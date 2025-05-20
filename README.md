# MA Thesis repo
## Филиппенко Илья, P4255

Чтобы воспроизвести

1. (Опционально) Добавить в файл .env параметры для itmo id: 
ITMO_CLIENT_ID
ITMO_CLIENT_SECRET
```shell
# 2. скачать модель с huggingface или поменять конфигурацию llama server
# --hf-repo itlwas/YandexGPT-5-Lite-8B-instruct-Q4_K_M-GGUF --hf-file yandexgpt-5-lite-8b-instruct-q4_k_m.gguf 
curl -L https://huggingface.co/itlwas/YandexGPT-5-Lite-8B-instruct-Q4_K_M-GGUF/resolve/main/yandexgpt-5-lite-8b-instruct-q4_k_m.gguf?download=true -o ./models/yandexgpt-5-lite-8b-instruct-q4_k_m.gguf

# 3. Довериться docker compose
docker compose up --build 
```