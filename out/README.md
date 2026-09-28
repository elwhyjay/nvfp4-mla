# out/ — 벤치마크 raw output

- `logs/`    `<시각>_<GPU>_<명령>.txt` — `bench.py::sh`로 돌린 원격 실행의 stdout+stderr 전부 (Modal 빌드 로그 포함될 수 있음)
- `ncu/`     `<스크립트>_<GPU>.ncu-rep` (GUI용) / `.txt` (`ncu --page details`) / `.log` (ncu 진행 출력)
- `results/` `baselines_<GPU>.csv` — baselines.py의 정리된 수치. 원본 텍스트 표는 같은 시각의 logs/ 파일에 있음

같은 파일이 Modal Volume `kernel-bench-cache`의 `results/`에도 남는다.

## logs/modal-apps/ — 서버 보관 로그 (사후 복구용)

`bench.py::sh` 도입 전 실행들을 `modal app logs <app-id>`로 받아 온 것. 컨테이너 stdout만 담겨 있어 로컬 tee 로그보다 짧다.
`_build-only`가 붙은 두 개는 ncu 프로파일 실행으로, 함수 출력을 capture해 로컬로 반환한 구조라 서버에는 이미지 빌드 로그만 남았다. 그 출력은 `ncu/`에 있다.
지난 실행은 `uv run modal app list`로 ID를 찾고 `uv run modal app logs <id>`로 다시 받을 수 있다.
