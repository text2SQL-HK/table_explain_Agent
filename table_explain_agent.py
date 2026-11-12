from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import oracledb
except ModuleNotFoundError:
    oracledb = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[assignment]

SYS_PROMPT = """
Describe the table and its columns using only the supplied [INFO] and [Context].
Use the column names, data types, authorized example values, and comments provided at runtime.
Do not invent database names, business rules, locations, or example data.
Return one JSON object mapping each column name to its description, with a "table" description.
"""


@dataclass
class Config:
    base_dir: Path
    run_date: str
    table_names_path: Path
    time_column_path: Path
    oracle_info_path: Path
    comments_db_path: Path
    context_path: Path
    metadata_path: Path
    output_path: Path
    oracle_client_lib_dir: Optional[str]
    sample_percent: int
    max_values_per_column: int
    openai_model: str
    openai_base_url: Optional[str]
    openai_api_key: str
    max_output_tokens: int
    reasoning_effort: str
    force_metadata: bool
    metadata_only: bool


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def normalized_run_date(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) < 8:
        raise ValueError(f"--date must include at least YYYYMMDD: {value}")
    return digits[:8]


def date_suffix(value: str) -> str:
    return normalized_run_date(value)[4:]


def parse_args() -> Config:
    base_dir = Path(__file__).resolve().parent
    today = datetime.now().strftime("%Y%m%d")
    parser = argparse.ArgumentParser(
        description="Build metadata and table descriptions in one unified directory."
    )
    parser.add_argument("--date", default=today, help="Run date (YYYYMMDD).")
    parser.add_argument("--table-names", default="input/table_names_train.json")
    parser.add_argument("--time-column", default="input/time_column.json")
    parser.add_argument("--oracle-info", default="input/ORACLE_INFO.json")
    parser.add_argument("--comments-db", default="input/comments.sqlite")
    parser.add_argument("--context", default="docs/context.txt")
    parser.add_argument("--metadata", default="output/metadata.json")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--oracle-client-lib-dir",
        default=os.getenv("ORACLE_CLIENT_LIB_DIR", ""),
    )
    parser.add_argument("--sample-percent", type=int, default=1)
    parser.add_argument("--max-values-per-column", type=int, default=4)
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "/workspace/models/gpt-oss-120b"),
    )
    parser.add_argument(
        "--openai-base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--openai-api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
    )
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--reasoning-effort", default="high", choices=["low", "medium", "high"])
    parser.add_argument("--force-metadata", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    run_date = normalized_run_date(args.date)
    default_output = f"output/tables_{date_suffix(run_date)}.json"
    output_path = resolve_path(base_dir, args.output or default_output)

    return Config(
        base_dir=base_dir,
        run_date=run_date,
        table_names_path=resolve_path(base_dir, args.table_names),
        time_column_path=resolve_path(base_dir, args.time_column),
        oracle_info_path=resolve_path(base_dir, args.oracle_info),
        comments_db_path=resolve_path(base_dir, args.comments_db),
        context_path=resolve_path(base_dir, args.context),
        metadata_path=resolve_path(base_dir, args.metadata),
        output_path=output_path,
        oracle_client_lib_dir=args.oracle_client_lib_dir or None,
        sample_percent=args.sample_percent,
        max_values_per_column=args.max_values_per_column,
        openai_model=args.model,
        openai_base_url=args.openai_base_url or None,
        openai_api_key=args.openai_api_key,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        force_metadata=args.force_metadata,
        metadata_only=args.metadata_only,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def create_dynamic_schema(input_dict: Dict[str, Optional[str]]) -> Dict[str, Any]:
    properties: Dict[str, Dict[str, str]] = {}
    required_fields: List[str] = []
    for key in input_dict:
        properties[key] = {"type": "string"}
        required_fields.append(key)
    properties["table"] = {"type": "string"}
    required_fields.append("table")
    return {
        "type": "object",
        "properties": properties,
        "required": required_fields,
        "additionalProperties": False,
    }


def split_owner_table(table_name: str) -> Tuple[Optional[str], str]:
    parts = table_name.split(".", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, parts[0]


def quote_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def table_ref(owner: Optional[str], table: str) -> str:
    if owner:
        return f"{quote_identifier(owner)}.{quote_identifier(table)}"
    return quote_identifier(table)


def format_column_type(
    data_type: str,
    data_length: Optional[int],
    data_precision: Optional[int],
    data_scale: Optional[int],
) -> str:
    if data_type == "VARCHAR2" and data_length is not None:
        return f"{data_type}({data_length})"
    if data_type == "NUMBER":
        if data_precision is None:
            return data_type
        if data_scale is None:
            return f"{data_type}({data_precision})"
        return f"{data_type}({data_precision},{data_scale})"
    return data_type


def normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if isinstance(value, date_cls):
        return value.strftime("%Y-%m-%d")
    return value


def date_preprocess(run_date: str, time_column: str) -> Tuple[str, str]:
    dt = datetime.strptime(run_date, "%Y%m%d")
    prev_month_last_day = dt.replace(day=1) - timedelta(days=1)
    prev_month_day = min(dt.day, prev_month_last_day.day)
    prev_month_dt = prev_month_last_day.replace(day=prev_month_day)

    start = prev_month_dt.strftime("%Y%m%d")
    end = dt.strftime("%Y%m%d")
    if "DTM" in time_column.upper():
        start += "000000"
        end += "000000"
    return start, end


def find_comment(cache_conn: sqlite3.Connection, column_name: str) -> Optional[str]:
    row = cache_conn.execute(
        "SELECT COMMENTS FROM COMMENTS WHERE COLUMN_NAME = ?",
        (column_name,),
    ).fetchone()
    if not row:
        return None
    return row[0]


def fetch_column_comments(
    conn: oracledb.Connection,
    owner: Optional[str],
    table: str,
) -> List[Tuple[str, Optional[str]]]:
    params: Dict[str, Any] = {"table_name": table.upper()}
    where = "TABLE_NAME = :table_name"
    if owner:
        where += " AND OWNER = :owner"
        params["owner"] = owner.upper()

    sql = f"""
        SELECT COLUMN_NAME, COMMENTS
        FROM SYS.ALL_COL_COMMENTS
        WHERE {where}
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [(row[0], row[1]) for row in cur.fetchall()]


def fetch_column_types(
    conn: oracledb.Connection,
    owner: Optional[str],
    table: str,
) -> Dict[str, str]:
    params: Dict[str, Any] = {"table_name": table.upper()}
    where = "TABLE_NAME = :table_name"
    if owner:
        where += " AND OWNER = :owner"
        params["owner"] = owner.upper()

    sql = f"""
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            DATA_LENGTH,
            DATA_PRECISION,
            DATA_SCALE
        FROM SYS.ALL_TAB_COLUMNS
        WHERE {where}
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    formatted: Dict[str, str] = {}
    for row in rows:
        formatted[row[0]] = format_column_type(row[1], row[2], row[3], row[4])
    return formatted


def fetch_empty_columns(
    conn: oracledb.Connection,
    owner: Optional[str],
    table: str,
) -> List[str]:
    params: Dict[str, Any] = {"table_name": table.upper()}
    where = "TABLE_NAME = :table_name AND NUM_DISTINCT = 0"
    if owner:
        where += " AND OWNER = :owner"
        params["owner"] = owner.upper()

    sql = f"""
        SELECT COLUMN_NAME
        FROM ALL_TAB_COL_STATISTICS
        WHERE {where}
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [row[0] for row in cur.fetchall()]


def fetch_sample_values(
    conn: oracledb.Connection,
    owner: Optional[str],
    table: str,
    column_name: str,
    time_column: str,
    run_date: str,
    sample_percent: int,
    max_values: int,
) -> List[Any]:
    table_name = table_ref(owner, table)
    col = quote_identifier(column_name)
    sample_clause = f"SAMPLE({sample_percent})" if sample_percent > 0 else ""

    params: Dict[str, Any] = {"limit": max_values}
    time_filter = ""
    if time_column:
        time_col = quote_identifier(time_column)
        t1, t2 = date_preprocess(run_date, time_column)
        params["t1"] = t1
        params["t2"] = t2
        time_filter = f"AND {time_col} BETWEEN :t1 AND :t2"

    sql = f"""
        SELECT *
        FROM (
            SELECT DISTINCT {col}
            FROM {table_name} {sample_clause}
            WHERE {col} IS NOT NULL
            {time_filter}
        )
        WHERE ROWNUM <= :limit
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        raw_values = [row[0] for row in cur.fetchall()]
    cleaned = [normalize_value(v) for v in raw_values if v != ""]
    return cleaned


def init_oracle_client(lib_dir: Optional[str]) -> None:
    if oracledb is None:
        raise ModuleNotFoundError("oracledb is not installed. Install it with `pip install oracledb`.")
    if not lib_dir:
        return
    try:
        oracledb.init_oracle_client(lib_dir=lib_dir)
    except Exception as exc:
        if "already been initialized" not in str(exc):
            raise


def connect_oracle(db_name: str, oracle_info: Dict[str, Any]) -> oracledb.Connection:
    if oracledb is None:
        raise ModuleNotFoundError("oracledb is not installed. Install it with `pip install oracledb`.")
    if db_name not in oracle_info:
        raise KeyError(f"Missing Oracle connection info for db_name: {db_name}")
    info = oracle_info[db_name]
    dsn = f"{info['ORACLE_HOST']}:{info['ORACLE_PORT']}/{info['ORACLE_SERVICE_NAME']}"
    return oracledb.connect(
        user=info["ORACLE_USER"],
        password=info["ORACLE_PASSWORD"],
        dsn=dsn,
    )


def build_time_column_map(time_json: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapped: Dict[str, List[str]] = {}
    for entry in time_json:
        mapped[entry["db_name"]] = entry.get("time_column", [])
    return mapped


def build_metadata(config: Config, db_json: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    oracle_info = load_json(config.oracle_info_path)
    time_json = load_json(config.time_column_path)
    time_map = build_time_column_map(time_json)

    init_oracle_client(config.oracle_client_lib_dir)
    cache_conn = sqlite3.connect(str(config.comments_db_path))
    db_info_list: List[Dict[str, Any]] = []
    try:
        for db_entry in db_json:
            db_name = db_entry["db_name"]
            table_names: List[str] = db_entry["table_names"]
            time_columns = time_map.get(db_name, [])
            if len(time_columns) < len(table_names):
                time_columns = time_columns + [""] * (len(table_names) - len(time_columns))

            print(f"[metadata] db={db_name}, table_count={len(table_names)}")
            conn = connect_oracle(db_name, oracle_info)
            db_info: List[Dict[str, Any]] = []
            try:
                for idx, full_table_name in enumerate(table_names):
                    time_column = time_columns[idx] if idx < len(time_columns) else ""
                    owner, table = split_owner_table(full_table_name)
                    print(f"  - table={full_table_name}, time_column={time_column}")

                    comments = fetch_column_comments(conn, owner, table)
                    comments_with_fallback: List[Tuple[str, Optional[str]]] = []
                    for column_name, comment in comments:
                        comments_with_fallback.append(
                            (column_name, comment if comment is not None else find_comment(cache_conn, column_name))
                        )

                    column_types = fetch_column_types(conn, owner, table)
                    empty_columns = set(fetch_empty_columns(conn, owner, table))
                    filtered_comments = [
                        item for item in comments_with_fallback if item[0] not in empty_columns
                    ]
                    for empty_col in empty_columns:
                        column_types.pop(empty_col, None)

                    schema = create_dynamic_schema(dict(filtered_comments))
                    column_info: List[Dict[str, Any]] = []
                    for column_name, comment in filtered_comments:
                        try:
                            values = fetch_sample_values(
                                conn=conn,
                                owner=owner,
                                table=table,
                                column_name=column_name,
                                time_column=time_column,
                                run_date=config.run_date,
                                sample_percent=config.sample_percent,
                                max_values=config.max_values_per_column,
                            )
                        except Exception as exc:
                            print(f"    [warn] failed to sample values for {full_table_name}.{column_name}: {exc}")
                            values = []
                        column_info.append(
                            {
                                "column_name": column_name,
                                "comment": comment,
                                "type": column_types.get(column_name, "UNKNOWN"),
                                "value": values,
                            }
                        )

                    db_info.append(
                        {
                            "table_name": full_table_name,
                            "column_info": column_info,
                            "schema": schema,
                        }
                    )
            finally:
                conn.close()

            db_info_list.append({"db_name": db_name, "db_info": db_info})
    finally:
        cache_conn.close()
    return db_info_list


def create_openai_client(config: Config) -> OpenAI:
    if OpenAI is None:
        raise ModuleNotFoundError("openai is not installed. Install it with `pip install openai`.")
    if config.openai_base_url:
        return OpenAI(base_url=config.openai_base_url, api_key=config.openai_api_key)
    return OpenAI(api_key=config.openai_api_key)


def explain_table(
    client: OpenAI,
    config: Config,
    context: str,
    table_name: str,
    column_info: List[Dict[str, Any]],
    schema: Dict[str, Any],
) -> Dict[str, str]:
    payload = json.dumps(column_info, ensure_ascii=False, indent=4)
    response = client.responses.create(
        model=config.openai_model,
        input=[
            {
                "role": "system",
                "content": SYS_PROMPT.format(
                    info=payload,
                    context=context,
                    table_name=table_name,
                ),
            },
            {"role": "user", "content": ""},
        ],
        max_output_tokens=config.max_output_tokens,
        text={
            "format": {
                "type": "json_schema",
                "name": "column_data",
                "schema": schema,
            }
        },
        stream=False,
        reasoning={"effort": config.reasoning_effort},
    )
    return json.loads(response.output_text)


def generate_final_outputs(
    config: Config,
    db_json: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    context = config.context_path.read_text(encoding="utf-8")
    client = create_openai_client(config)
    metadata_map = {item["db_name"]: item for item in metadata}
    final_jsons: List[Dict[str, Any]] = []

    for db_entry in db_json:
        db_name = db_entry["db_name"]
        if db_name not in metadata_map:
            raise KeyError(f"metadata has no db_name={db_name}")
        db_meta = metadata_map[db_name]

        table_names_original: List[str] = []
        table_names_desc: List[str] = []
        column_names_original: List[List[Any]] = [[-1, "*"]]
        column_names: List[List[Any]] = [[-1, "*"]]
        column_types: List[str] = ["text"]
        column_values: List[List[Any]] = [["hello_world"]]

        db_tables = db_meta["db_info"]
        for table_idx, table_meta in enumerate(db_tables):
            table_name = table_meta["table_name"]
            column_info = table_meta["column_info"]
            schema = table_meta["schema"]

            print(f"[explain] db={db_name}, table={table_name}")
            result = explain_table(
                client=client,
                config=config,
                context=context,
                table_name=table_name,
                column_info=column_info,
                schema=schema,
            )

            table_names_original.append(table_name)
            table_names_desc.append(result.get("table", ""))
            for col in column_info:
                column_name = col["column_name"]
                column_names_original.append([table_idx, column_name])
                column_names.append([table_idx, result.get(column_name, col.get("comment") or "")])
                column_types.append(col["type"])
                column_values.append(col.get("value", []))

        final_jsons.append(
            {
                "db_id": db_name,
                "table_names_original": table_names_original,
                "table_names": table_names_desc,
                "column_names_original": column_names_original,
                "column_names": column_names,
                "column_types": column_types,
                "primary_keys": [],
                "foreign_keys": [],
                "values": column_values,
            }
        )

    return final_jsons


def main() -> None:
    config = parse_args()
    db_json = load_json(config.table_names_path)

    if config.force_metadata or not config.metadata_path.exists():
        print(f"[run] building metadata -> {config.metadata_path}")
        metadata = build_metadata(config, db_json)
        save_json(config.metadata_path, metadata)
    else:
        print(f"[run] reusing metadata -> {config.metadata_path}")
        metadata = load_json(config.metadata_path)

    if config.metadata_only:
        print("[run] metadata-only mode complete")
        return

    print(f"[run] generating output -> {config.output_path}")
    final_jsons = generate_final_outputs(config, db_json, metadata)
    save_json(config.output_path, final_jsons)
    print("[run] complete")


if __name__ == "__main__":
    main()
