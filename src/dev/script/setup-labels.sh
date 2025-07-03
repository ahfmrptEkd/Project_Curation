#!/bin/bash
# setup-labels.sh

# 우선순위 라벨
gh label create "priority: critical" --color "d73a4a" --description "최우선 처리 필요"
gh label create "priority: high" --color "ff6b35" --description "높은 우선순위"
gh label create "priority: medium" --color "ffcc00" --description "보통 우선순위"
gh label create "priority: low" --color "0e8a16" --description "낮은 우선순위"

# 타입 라벨
gh label create "type: feature" --color "84b6eb" --description "새로운 기능"
gh label create "type: bug" --color "d73a4a" --description "버그 수정"
gh label create "type: enhancement" --color "a2eeef" --description "기능 개선"
gh label create "type: documentation" --color "0075ca" --description "문서화"

# 영역 라벨
gh label create "area: cold-start" --color "b60205" --description "Cold Start Problem"
gh label create "area: mcp" --color "1d76db" --description "MCP 통합"
gh label create "area: rag" --color "0e8a16" --description "RAG 시스템"
gh label create "area: prompt" --color "5319e7" --description "프롬프트 엔지니어링"
gh label create "area: ui" --color "f9d0c4" --description "사용자 인터페이스"
gh label create "area: deployment" --color "000000" --description "배포 관련"

# 난이도 라벨
gh label create "difficulty: easy" --color "c2e0c6" --description "1-2일 소요"
gh label create "difficulty: medium" --color "fef2c0" --description "3-5일 소요"
gh label create "difficulty: hard" --color "f9c2c2" --description "1주 이상 소요"
