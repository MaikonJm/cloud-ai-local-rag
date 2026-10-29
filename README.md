# Cloud AI v0.1

Assistente de inteligência artificial local com arquitetura RAG (Retrieval-Augmented Generation), ingestão de documentos, OCR, classificação automática e interface web.

## Arquitetura

- **Open WebUI:** interface de conversação.
- **FastAPI:** Knowledge API, autenticação, ingestão e consultas RAG.
- **Ollama:** execução local dos modelos de linguagem e embeddings.
- **Qdrant:** armazenamento e recuperação vetorial.
- **SQLite:** metadados e estados dos documentos.
- **Tesseract OCR:** extração de texto de PDFs digitalizados.
- **Docker Compose:** orquestração dos serviços.

### Fluxo de ingestão

1. Documento enviado à Knowledge API.
2. Arquivo armazenado e registrado no SQLite.
3. Extração de texto, com OCR quando necessário.
4. Classificação automática por categoria.
5. Aprovação automática ou encaminhamento para revisão.
6. Geração de embeddings e indexação no Qdrant.

### Fluxo de consulta

1. Usuário seleciona o modelo `cloud-ai` no Open WebUI.
2. A Knowledge API recebe a pergunta.
3. A pergunta é convertida em embedding.
4. O Qdrant recupera trechos semanticamente relacionados.
5. O modelo local gera uma resposta com base no contexto.
6. A API retorna a resposta e as fontes recuperadas.

## Ambiente utilizado no desenvolvimento

- Host: Windows 11.
- Virtualização: VMware Workstation.
- Sistema convidado: Ubuntu Server 26.04.1 LTS.
- VM: 16 GB de RAM, 8 vCPUs e disco virtual de 120 GB.
- Docker Engine e Docker Compose.
- Inferência local por CPU, sem GPU passthrough.

## Modelos

- `qwen2.5:3b`: geração de respostas RAG.
- `nomic-embed-text`: geração de embeddings.
- `qwen3:4b`: modelo adicional utilizado durante experimentos.

## Instalação

Requisitos: Linux com Docker Engine, Docker Compose e recursos suficientes para executar os modelos locais.

```bash
cp .env.example .env
sed -i "s/REPLACE_WITH_RANDOM_SECRET/$(openssl rand -hex 32)/;s/REPLACE_WITH_RANDOM_TOKEN/$(openssl rand -hex 32)/" .env
chmod 600 .env

docker compose up -d --build

docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull nomic-embed-text


curl http://127.0.0.1:8000/health
TOKEN=$(sed -n 's/^API_TOKEN=//p' .env | tr -d '\r"')

curl -X POST http://127.0.0.1:8000/upload \
  -H "Authorization: Bearer ${TOKEN}" \
  -F "file=@examples/aws-sqs-test.md"
curl http://127.0.0.1:8000/documents/DOCUMENT_ID \
  -H "Authorization: Bearer ${TOKEN}"
