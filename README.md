Модель paraphrase-multilingual-MiniLM-L12-v2 можно скачать с Hugging Face:

Страница модели:
https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Для работы понадобятся три файла:

config.json – конфигурация архитектуры модели.
tokenizer.json – токенизатор.
model.safetensors – веса модели (если в репозитории лежит только pytorch_model.bin, его можно конвертировать в safetensors или использовать Candle напрямую с bin-файлом, но safetensors предпочтительнее).

Как скачать
Вариант 1: через браузер
Перейдите на страницу модели, вкладка Files, и скачайте указанные файлы вручную.

Вариант 2: через git lfs
git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
После клонирования в папке будут все файлы.
