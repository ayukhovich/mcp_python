"""
MCP Client с поддержкой SSE и интеграцией с LLM (Ollama)
Клиент подключается к MCP серверу через SSE, получает список доступных инструментов,
использует LLM для определения нужного инструмента и вызывает его.
"""

# Стандартные библиотеки Python
import asyncio  # Для асинхронного программирования
import datetime
import json  # Для работы с JSON (парсинг ответов LLM)
import logging  # Для логирования работы клиента
import os  # Для работы с файловой системой (создание директорий)
import sys  # Для доступа к системным параметрам (stdout)
import re  # Для регулярных выражений (извлечение JSON из ответа)

# Сторонние библиотеки
from colorama import init, Fore, Style  # Для цветного вывода в консоль (Windows совместимость)
from openai import OpenAI  # Клиент для работы с Ollama (совместимый с OpenAI API)
from dotenv import load_dotenv  # Для загрузки переменных окружения из .env файла
import httpx  # HTTP клиент (используется для патчинга запросов)

# MCP библиотеки
from mcp import ClientSession  # Основной класс для работы с MCP сессией
from mcp.client.sse import sse_client  # Клиент для SSE подключения к серверу

# Инициализация colorama для корректного отображения цветов в Windows
# autoreset=True автоматически сбрасывает цвет после каждого print
init(autoreset=True)

# Загружаем переменные окружения из .env файла (например, API ключи)
load_dotenv()

# Сохраняем оригинальный метод request для последующего патчинга
# Это нужно чтобы корректно обрабатывать редиректы (307 → /messages/)
_orig_request = httpx.AsyncClient.request

# Создаем директорию для логов, если её нет
# __file__ - путь к текущему файлу, os.path.dirname - директория файла
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)  # exist_ok=True не вызывает ошибку если директория уже есть
log_file = os.path.join(log_dir, 'client.log')

# Настройка системы логирования
# level=logging.DEBUG - логируем всё от DEBUG и выше
# format - формат сообщения: время [уровень] источник: сообщение
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] CLIENT: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Вывод в консоль
        logging.FileHandler(log_file, encoding='utf-8')  # Запись в файл с поддержкой Unicode
    ]
)
# Создаем логгер с именем "mcp-client"
logger = logging.getLogger("mcp-client")

# Добавляем кастомный уровень логирования SUCCESS (между INFO и WARNING)
# Это позволит выделять успешные операции
logging.SUCCESS = 25  # 25 - между WARNING (30) и INFO (20)
logging.MCP_REQUEST = 21  # Для MCP запросов (чуть выше INFO)
logging.MCP_RESPONSE = 22  # Для MCP ответов (чуть выше INFO)

logging.addLevelName(logging.SUCCESS, 'SUCCESS')
logging.addLevelName(logging.MCP_REQUEST, 'MCP_REQUEST')
logging.addLevelName(logging.MCP_RESPONSE, 'MCP_RESPONSE')

# Добавляем методы к логгеру
setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))
setattr(logger, 'mcp_request', lambda message, *args: logger._log(logging.MCP_REQUEST, message, args))
setattr(logger, 'mcp_response', lambda message, *args: logger._log(logging.MCP_RESPONSE, message, args))


async def _patched_request(self, method, url, *args, **kwargs):
    """
    Патч для httpx.AsyncClient.request, который гарантирует,
    что follow_redirects=True для всех запросов.

    Это необходимо потому что MCP сервер может возвращать статус 307 (Temporary Redirect)
    при перенаправлении на /messages/ эндпоинт.

    Аргументы:
        self: экземпляр клиента
        method: HTTP метод (GET, POST и т.д.)
        url: URL запроса
        *args, **kwargs: остальные аргументы оригинального метода

    Возвращает:
        Результат оригинального метода request с follow_redirects=True
    """
    # Устанавливаем follow_redirects=True по умолчанию, если не указано иное
    kwargs.setdefault("follow_redirects", True)
    # Вызываем оригинальный метод с обновленными kwargs
    return await _orig_request(self, method, url, *args, **kwargs)


# Применяем патч: заменяем оригинальный метод на наш patched_request
httpx.AsyncClient.request = _patched_request


def llm_client(message: str) -> str:
    """
    Отправляет сообщение в локальную LLM модель через Ollama.

    Использует OpenAI-совместимый клиент для подключения к Ollama,
    которая должна быть запущена локально на порту 11431.

    Аргументы:
        message (str): промпт для отправки в модель

    Возвращает:
        str: текстовый ответ от модели
    """
    logger.info(f"{Fore.GREEN}📤 Sending message to LLM: {message}")

    # Создаем клиент OpenAI, но подключаемся к локальному серверу Ollama
    # base_url - адрес Ollama сервера (по умолчанию 11434, но у вас 11431)
    # api_key - обязательный параметр для OpenAI клиента, но Ollama его игнорирует
    client = OpenAI(
        base_url='http://localhost:11431/v1/',
        api_key='ollama'
    )

    # Отправляем запрос к модели
    # model - имя модели в Ollama (должна быть предварительно загружена)
    # temperature=0.1 - низкая температура для более детерминированных ответов
    completion = client.chat.completions.create(
        model="gemma3:4b-it-q4_K_M",
        messages=[
            # Системное сообщение задает поведение модели
            {"role": "system",
             "content": "You are an intelligent Assistant. You will execute tasks as instructed. Always respond with valid JSON only, no explanations."},
            # Сообщение пользователя - наш промпт
            {"role": "user", "content": message},
        ],
        temperature=0.1
    )

    # Извлекаем текст ответа из структуры completion
    result = completion.choices[0].message.content
    logger.info(f"{Fore.YELLOW}📥 Raw LLM response: {result}")
    return result


def get_prompt_to_identify_tool_and_arguments(query: str, tools) -> str:
    """
    Формирует промпт для LLM, который описывает доступные инструменты
    и просит выбрать подходящий для ответа на вопрос пользователя.

    Аргументы:
        query (str): вопрос пользователя
        tools: объект с описанием доступных инструментов от MCP сервера

    Возвращает:
        str: сформированный промпт для отправки в LLM
    """
    # Формируем описание каждого инструмента: имя, описание, схема входных данных
    tools_description = "\n".join([
        f"{tool.name}: {tool.description}, Input schema: {tool.inputSchema}"
        for tool in tools.tools
    ])

    # Возвращаем многострочный промпт с четкими инструкциями
    return (
        "You are a helpful assistant with access to these tools:\n\n"
        f"{tools_description}\n"
        "Choose the appropriate tool based on the user's question.\n"
        f"User's Question: {query}\n"
        "If no tool is needed, reply directly with a helpful response.\n\n"
        "IMPORTANT: When you need to use a tool, you must ONLY respond with "
        "the exact JSON object format below, nothing else - no explanations, no markdown:\n"
        "EXAMPLE RESPONSE:\n"
        '{\n'
        '    "tool": "time_tool",\n'
        '    "arguments": {\n'
        '        "input_timezone": "Asia/Kolkata"\n'
        '    }\n'
        '}\n\n'
    )


async def log_mcp_request_response(func):
    """
    Декоратор для логирования MCP запросов и ответов.

    Оборачивает методы сессии и логирует:
    - Время отправки запроса
    - Название метода
    - Параметры запроса
    - Время получения ответа
    - Полный ответ от сервера

    Аргументы:
        func: асинхронная функция для оборачивания

    Возвращает:
        обернутую функцию с логированием
    """

    async def wrapper(*args, **kwargs):
        # Получаем имя метода из первого аргумента (self)
        method_name = func.__name__

        # Формируем сообщение о запросе
        request_msg = f"🔷 MCP Request [{method_name}]"
        if kwargs:
            # Если есть аргументы, добавляем их в лог
            kwargs_str = json.dumps(kwargs, indent=2, ensure_ascii=False)
            request_msg += f"\nArguments:\n{kwargs_str}"
        elif len(args) > 1:  # первый аргумент self, остальные позиционные
            args_str = json.dumps(args[1:], indent=2, ensure_ascii=False)
            request_msg += f"\nArguments:\n{args_str}"

        # Логируем запрос
        logger.mcp_request(f"{Fore.CYAN}{request_msg}")
        logger.mcp_request(f"{Fore.CYAN}⏰ Request time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

        # Засекаем время начала
        start_time = datetime.now()

        try:
            # Выполняем оригинальный метод
            result = await func(*args, **kwargs)

            # Вычисляем время выполнения
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() * 1000  # в миллисекундах

            # Формируем сообщение об ответе
            response_msg = f"🔶 MCP Response [{method_name}]"
            response_msg += f"\n⏱️ Duration: {duration:.2f}ms"
            response_msg += f"\n⏰ Response time: {end_time.strftime('%H:%M:%S.%f')[:-3]}"

            # Добавляем содержимое ответа в зависимости от типа
            if hasattr(result, 'content') and result.content:
                if hasattr(result.content[0], 'text'):
                    response_msg += f"\n📄 Content: {result.content[0].text}"
                else:
                    response_msg += f"\n📄 Content: {str(result.content[0])}"
            elif isinstance(result, dict):
                response_msg += f"\n📄 Response:\n{json.dumps(result, indent=2, ensure_ascii=False)}"
            elif hasattr(result, 'tools') and result.tools:
                # Для ответа list_tools
                tools_list = [f"{tool.name}: {tool.description}" for tool in result.tools]
                response_msg += f"\n🔧 Tools available:\n" + "\n".join(tools_list)
            else:
                response_msg += f"\n📄 Response: {str(result)}"

            # Логируем ответ
            logger.mcp_response(f"{Fore.MAGENTA}{response_msg}")

            return result

        except Exception as e:
            # Логируем ошибку
            error_time = datetime.now()
            duration = (error_time - start_time).total_seconds() * 1000
            logger.error(f"{Fore.RED}❌ MCP Error [{method_name}] after {duration:.2f}ms: {str(e)}")
            raise

    return wrapper


class LoggedClientSession(ClientSession):
    """
    Обертка над ClientSession с логированием всех MCP вызовов.

    Переопределяет основные методы для добавления логирования:
    - initialize
    - list_tools
    - call_tool
    """

    @log_mcp_request_response
    async def initialize(self):
        """Инициализация сессии с логированием"""
        return await super().initialize()

    @log_mcp_request_response
    async def list_tools(self):
        """Получение списка инструментов с логированием"""
        return await super().list_tools()

    @log_mcp_request_response
    async def call_tool(self, name: str, arguments: dict = None):
        """
        Вызов инструмента с логированием

        Аргументы:
            name: имя инструмента
            arguments: словарь аргументов для инструмента
        """
        return await super().call_tool(name, arguments)


async def main(query: str):
    """
    Основная асинхронная функция обработки одного запроса.

    Последовательность действий:
    1. Подключение к MCP серверу через SSE
    2. Инициализация сессии
    3. Получение списка доступных инструментов
    4. Формирование промпта и отправка в LLM
    5. Парсинг JSON ответа от LLM
    6. Вызов выбранного инструмента на сервере
    7. Вывод результата

    Аргументы:
        query (str): вопрос пользователя для обработки
    """
    # Визуальное разделение запросов в логах
    logger.info(f"{Fore.CYAN}{'=' * 60}")
    logger.info(f"{Fore.CYAN}🔍 Processing query: {query}")
    logger.info(f"{Fore.CYAN}{'=' * 60}")

    # URL SSE эндпоинта MCP сервера
    # Сервер должен быть запущен и доступен по этому адресу
    sse_url = "http://localhost:8000/sse"

    # 1) Открываем SSE соединение с сервером
    # sse_client возвращает кортеж (входной поток, выходной поток)
    # async with гарантирует правильное закрытие соединения даже при ошибках
    async with sse_client(url=sse_url) as (in_stream, out_stream):
        # 2) Создаем MCP сессию поверх потоков SSE
        async with ClientSession(in_stream, out_stream) as session:
            # 3) Инициализируем сессию - обязательный шаг перед любой работой
            # Получаем информацию о сервере (имя, версия)
            info = await session.initialize()
            logger.info(f"{Fore.GREEN}✅ Connected to {info.serverInfo.name} v{info.serverInfo.version}")

            # 4) Получаем список доступных инструментов от сервера
            tools = await session.list_tools()
            logger.info(f"{Fore.MAGENTA}🔧 Available tools: {[tool.name for tool in tools.tools]}")

            # Детальное логирование каждого инструмента
            for tool in tools.tools:
                logger.debug(f"{Fore.MAGENTA}  • {tool.name}: {tool.description}")
                logger.debug(f"{Fore.MAGENTA}    Input schema: {tool.inputSchema}")

            # 5) Формируем промпт для LLM на основе вопроса и доступных инструментов
            prompt = get_prompt_to_identify_tool_and_arguments(query, tools)
            logger.info(f"{Fore.BLUE}{'=' * 40}")
            logger.info(f"{Fore.BLUE}📝 Prompt sent to LLM:")
            logger.info(f"{Fore.BLUE}{prompt}")
            logger.info(f"{Fore.BLUE}{'=' * 40}")

            # 6) Отправляем промпт в LLM и получаем ответ
            response = llm_client(prompt)

            logger.info(f"{Fore.RED}{'=' * 40}")
            logger.info(f"{Fore.RED}📥 Raw response from LLM:")
            logger.info(f"{Fore.RED}{response}")
            logger.info(f"{Fore.RED}{'=' * 40}")

            # 7) Пытаемся распарсить ответ как JSON
            try:
                # Используем регулярное выражение для поиска JSON в ответе
                # Это на случай, если модель добавила пояснения до или после JSON
                # re.DOTALL позволяет точке соответствовать также символам новой строки
                json_match = re.search(r'\{.*\}', response, re.DOTALL)

                if json_match:
                    # Извлекаем найденный JSON как строку
                    json_str = json_match.group()
                    logger.info(f"{Fore.YELLOW}🔍 Extracted JSON: {json_str}")
                    # Парсим строку в Python объект (словарь)
                    tool_call = json.loads(json_str)
                else:
                    # Если JSON не найден, пробуем распарсить весь ответ
                    # Это вызовет JSONDecodeError если ответ не является валидным JSON
                    tool_call = json.loads(response)

                # Логируем успешно распарсенный вызов инструмента
                logger.info(f"{Fore.GREEN}✅ Successfully parsed tool call:")
                logger.info(f"{Fore.GREEN}  🛠️ Tool: {tool_call['tool']}")
                logger.info(f"{Fore.GREEN}  📊 Arguments: {tool_call['arguments']}")
                logger.info(f"{Fore.GREEN}{'=' * 40}")

                # 8) Вызываем выбранный инструмент на MCP сервере
                # Передаем имя инструмента и аргументы
                logger.info(f"{Fore.YELLOW}🔄 Calling tool '{tool_call['tool']}' on MCP server...")
                result = await session.call_tool(tool_call["tool"], arguments=tool_call["arguments"])

                # 9) Извлекаем текст ответа из результата
                # Структура result может быть разной, поэтому делаем проверки
                if hasattr(result, 'content') and result.content:
                    # Проверяем есть ли атрибут text у первого элемента content
                    if hasattr(result.content[0], 'text'):
                        tool_response = result.content[0].text
                    else:
                        # Если нет text, преобразуем весь объект в строку
                        tool_response = str(result.content[0])
                else:
                    # Если нет content, преобразуем весь result в строку
                    tool_response = str(result)

                # Логируем успешное выполнение
                logger.success(f"{Fore.GREEN}✨ User query: {query}")
                logger.success(f"{Fore.GREEN}✨ Tool Response: {tool_response}")

                # Выводим результат пользователю
                print(f"\n{Fore.GREEN}{'⭐' * 30}")
                print(f"{Fore.GREEN}Result: {tool_response}")
                print(f"{Fore.GREEN}{'⭐' * 30}\n")

            except json.JSONDecodeError as e:
                # Ошибка парсинга JSON - модель вернула невалидный JSON
                logger.error(f"{Fore.RED}❌ JSONDecodeError: Failed to parse LLM response: {e}")
                logger.error(f"{Fore.RED}  Raw response was: {response}")
                print(f"\n{Fore.RED}Error: Could not parse LLM response as JSON. Response was: {response}\n")

            except KeyError as e:
                # Ошибка отсутствия ключа - JSON не содержит ожидаемых полей 'tool' или 'arguments'
                logger.error(f"{Fore.RED}❌ KeyError: Missing expected key in tool call: {e}")
                logger.error(
                    f"{Fore.RED}  Tool call structure: {tool_call if 'tool_call' in locals() else 'Not parsed'}")
                print(f"\n{Fore.RED}Error: Invalid tool call structure. Missing key: {e}\n")

            except Exception as e:
                # Любая другая непредвиденная ошибка
                logger.error(f"{Fore.RED}❌ Unexpected error: {type(e).__name__}: {e}")
                print(f"\n{Fore.RED}Error: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    """
    Точка входа в программу.
    Запускает обработку списка запросов последовательно.
    """
    # Список тестовых запросов
    queries = [
        "What is the time in Bengaluru?",  # Запрос времени
        "What is the weather like right now in Dubai?"  # Запрос погоды
    ]

    # Визуальное начало работы
    logger.info(f"{Fore.CYAN}{'🚀' * 30}")
    logger.info(f"{Fore.CYAN}Starting MCP Client with {len(queries)} queries")
    logger.info(f"{Fore.CYAN}{'🚀' * 30}")

    # Обрабатываем каждый запрос по очереди
    for i, query in enumerate(queries, 1):
        logger.info(f"{Fore.YELLOW}\n{'─' * 50}")
        logger.info(f"{Fore.YELLOW}📋 Processing query {i}/{len(queries)}")
        logger.info(f"{Fore.YELLOW}{'─' * 50}")
        try:
            # Запускаем асинхронную функцию main для текущего запроса
            # asyncio.run() создает новый event loop для каждой итерации
            asyncio.run(main(query))
        except Exception as e:
            # Ловим фатальные ошибки, которые не были обработаны внутри main
            logger.error(f"{Fore.RED}💥 Fatal error in main loop: {e}")
            print(f"{Fore.RED}Fatal error: {e}")

        # Если это не последний запрос, делаем паузу перед следующим
        if i < len(queries):
            logger.info(f"{Fore.YELLOW}⏱️ Waiting 1 second before next query...")
            # Небольшая пауза чтобы не перегружать сервер
            asyncio.run(asyncio.sleep(1))

    # Визуальное завершение работы
    logger.info(f"{Fore.CYAN}{'🏁' * 30}")
    logger.info(f"{Fore.CYAN}MCP Client finished")
    logger.info(f"{Fore.CYAN}{'🏁' * 30}")