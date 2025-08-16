import os

from dotenv import load_dotenv

load_dotenv()


URLS_TO_PARSE = [
    "https://eora.ru/cases/promyshlennaya-bezopasnost",
    "https://eora.ru/cases/lamoda-systema-segmentacii-i-poiska-po-pohozhey-odezhde",
    "https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/karas-golosovoy-assistent",
    "https://eora.ru/cases/assistenty-dlya-gorodov",
    "https://eora.ru/cases/avtomatizaciya-v-promyshlennosti/chemrar-raspoznovanie-molekul",
    "https://eora.ru/cases/zeptolab-skazki-pro-amnyama-dlya-sberbox",
    "https://eora.ru/cases/goosegaming-algoritm-dlya-ocenki-igrokov",
    "https://eora.ru/cases/dodo-pizza-robot-analitik-otzyvov",
    "https://eora.ru/cases/ifarm-nejroset-dlya-ferm",
    "https://eora.ru/cases/zhivibezstraha-navyk-dlya-proverki-rodinok",
    "https://eora.ru/cases/sportrecs-nejroset-operator-sportivnyh-translyacij",
    "https://eora.ru/cases/avon-chat-bot-dlya-zhenshchin",
    "https://eora.ru/cases/navyki-dlya-golosovyh-assistentov/navyk-dlya-proverki-loterejnyh-biletov",
    "https://eora.ru/cases/computer-vision/iss-analiz-foto-avtomobilej",
    "https://eora.ru/cases/purina-master-bot",
    "https://eora.ru/cases/skinclub-algoritm-dlya-ocenki-veroyatnostej",
    "https://eora.ru/cases/skolkovo-chat-bot-dlya-startapov-i-investorov",
    "https://eora.ru/cases/purina-podbor-korma-dlya-sobaki",
    "https://eora.ru/cases/purina-navyk-viktorina",
    "https://eora.ru/cases/dodo-pizza-pilot-po-avtomatizacii-kontakt-centra",
    "https://eora.ru/cases/dodo-pizza-avtomatizaciya-kontakt-centra",
    "https://eora.ru/cases/icl-bot-sufler-dlya-kontakt-centra",
    "https://eora.ru/cases/s7-navyk-dlya-podbora-aviabiletov",
    "https://eora.ru/cases/workeat-whatsapp-bot",
    "https://eora.ru/cases/absolyut-strahovanie-navyk-dlya-raschyota-strahovki",
    "https://eora.ru/cases/kazanexpress-poisk-tovarov-po-foto",
    "https://eora.ru/cases/kazanexpress-sistema-rekomendacij-na-sajte",
    "https://eora.ru/cases/intels-proverka-logotipa-na-plagiat",
    "https://eora.ru/cases/karcher-viktorina-s-voprosami-pro-uborku",
    "https://eora.ru/cases/chat-boty/purina-friskies-chat-bot-na-sajte",
    "https://eora.ru/cases/nejroset-segmentaciya-video",
    "https://eora.ru/cases/chat-boty/essa-nejroset-dlya-generacii-rolikov",
    "https://eora.ru/cases/qiwi-poisk-anomalij",
    "https://eora.ru/cases/frisbi-nejroset-dlya-raspoznavaniya-pokazanij-schetchikov",
    "https://eora.ru/cases/skazki-dlya-gugl-assistenta",
    "https://eora.ru/cases/chat-boty/hr-bot-dlya-magnit-kotoriy-priglashaet-na-sobesedovanie",
]

INDEX_DIR = os.getenv("INDEX_DIR", "vector_storage")
HNSW_INDEX_PATH = os.getenv(
    "HNSW_INDEX_PATH", os.path.join(INDEX_DIR, "hnsw_index.bin")
)
METADATA_PATH = os.getenv("METADATA_PATH", os.path.join(INDEX_DIR, "metadata.json"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDINGS_DIM = os.getenv("EMBEDDINGS_DIM", 1536)
EMBEDDING_CACHE_PATH = os.getenv(
    "EMBEDDING_CACHE_PATH", os.path.join(INDEX_DIR, "embedding_cache.json")
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1/")

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
}

if not OPENAI_API_KEY:
    raise ValueError("Необходимо установить переменную окружения OPENAI_API_KEY")
