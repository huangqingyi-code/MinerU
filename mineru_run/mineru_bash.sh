
# 构建容器
docker build -t mineru:latest -f Dockerfile .

# 部署
docker compose -f compose.yaml --profile api up -d


# 请求
curl -X POST http://127.0.0.1:8000/file_parse \
  -F "files=@/mnt/data/qyhuang/agentic_rag/PageIndex/examples/documents/attention-residuals.pdf" \
  -F "backend=vlm-auto-engine" \
  -F "formula_enable=true" \
  -F "table_enable=true" \
  -F "image_analysis=true" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_content_list=true" \
  -F "return_images=true" \
  -F "return_original_file=true" \
  -F "response_format_zip=true" \
  --output result.zip