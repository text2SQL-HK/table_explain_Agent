# table_explain_agent_unified

`table_explain_agent.py` 실행에 필요한 로직/입력/산출물 경로를 이 디렉토리 기준으로 통일한 버전입니다.

## 디렉토리 구조

- `table_explain_agent.py`: 메타데이터 생성 + LLM 설명 생성 메인 스크립트
- `run.sh`: 실행 래퍼
- `input/`
  - `ORACLE_INFO.json`
  - `table_names_train.json`
  - `time_column.json`
  - `comments.sqlite`
- `docs/context.txt`
- `output/`
  - `metadata.json` (중간 산출물)
  - `tables_*.json` (최종 결과)

## 실행

```bash
cd /tmp/workspace/table_explain_agent_unified
bash run.sh --date 20251219
```

## 자주 쓰는 옵션

- `--force-metadata`: 기존 `output/metadata.json`을 무시하고 다시 생성
- `--metadata-only`: 메타데이터만 생성하고 종료
- `--output output/my_result.json`: 최종 결과 파일명 지정
- `--model`, `--openai-base-url`, `--openai-api-key`: LLM 설정 오버라이드
