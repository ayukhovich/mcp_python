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
import re  # Для регулярных выражений (извлечение JSON из ответа)
import sys  # Для доступа к системным параметрам (stdout)

import httpx  # HTTP клиент (используется для патчинга запросов)
# Сторонние библиотеки
from colorama import init, Fore  # Для цветного вывода в консоль (Windows совместимость)
from dotenv import load_dotenv  # Для загрузки переменных окружения из .env файла
# MCP библиотеки
from mcp import ClientSession  # Основной класс для работы с MCP сессией
from mcp.client.sse import sse_client  # Клиент для SSE подключения к серверу
from openai import OpenAI  # Клиент для работы с Ollama (совместимый с OpenAI API)

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
    tool_list = list(tools.tools)
    
    tools_description = "\n".join([
        f"{tool.name}: {tool.description}, Input schema: {json.dumps(tool.inputSchema)}"
        for tool in tool_list
    ])

    # Формируем примеры динамически на основе ВСЕХ инструментов
    single_examples = []
    multiple_examples = []
    
    for tool in tool_list:
        args = {}
        if tool.inputSchema:
            props = tool.inputSchema.get('properties', {})
            for prop_name, prop_info in props.items():
                prop_type = prop_info.get('type', 'string')
                if prop_type == 'integer':
                    args[prop_name] = 1
                elif prop_type == 'boolean':
                    args[prop_name] = True
                else:
                    args[prop_name] = f"<{prop_name}>"
        
        if args:
            example = {"tool": tool.name, "arguments": args}
            single_examples.append(example)
            if len(multiple_examples) < 2:
                multiple_examples.append(example)

    # Формируем финальные примеры
    if single_examples:
        example_single = json.dumps(single_examples[0])
        example_multiple = json.dumps(multiple_examples[:2]) if len(multiple_examples) >= 2 else example_single
    else:
        example_single = '{"tool": "tool_name", "arguments": {}}'
        example_multiple = example_single

    return (
        "You are a helpful assistant with access to these tools:\n\n"
        f"{tools_description}\n\n"
        f"User question: {query}\n\n"
        "RULES:\n"
        "- You MUST use tools to get factual information (time, weather, etc.)\n"
        "- NEVER answer directly - always use tools\n"
        "- If question needs multiple facts, use JSON array with ALL needed tools\n"
        "- Return ONLY JSON, no explanations\n\n"
        f"Example (multiple tools):\n{example_multiple}\n\n"
        f"Example (single tool):\n{example_single}\n\n"
        "Response:"
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
                json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', response)

                if json_match:
                    json_str = json_match.group()
                    logger.info(f"Extracted JSON: {json_str}")
                    tool_calls = json.loads(json_str)
                else:
                    tool_calls = json.loads(response)

                # Обрабатываем случай когда LLM вернул {"response": "..."}
                if isinstance(tool_calls, dict):
                    if "response" in tool_calls:
                        print(f"\n{'=' * 40}")
                        print(f"Result: {tool_calls['response']}")
                        print(f"{'=' * 40}\n")
                    elif "tool" in tool_calls and "arguments" in tool_calls:
                        tool_calls = [tool_calls]
                    else:
                        raise ValueError("Invalid tool call format")
                elif isinstance(tool_calls, list):
                    valid_calls = [tc for tc in tool_calls if isinstance(tc, dict) and "tool" in tc and "arguments" in tc]
                    if not valid_calls:
                        raise ValueError("No valid tool calls found")
                    tool_calls = valid_calls
                else:
                    raise ValueError("Invalid tool call format")

                # Вызываем все инструменты
                results = []
                for tool_call in tool_calls:
                    logger.info(f"Calling tool: {tool_call['tool']}")
                    result = await session.call_tool(tool_call["tool"], arguments=tool_call["arguments"])

                    if hasattr(result, 'content') and result.content:
                        if hasattr(result.content[0], 'text'):
                            tool_response = result.content[0].text
                        else:
                            tool_response = str(result.content[0])
                    else:
                        tool_response = str(result)

                    results.append(f"{tool_call['tool']}: {tool_response}")

                tool_response = "\n".join(results)

                # Если несколько инструментов - отправляем в LLM для красивого ответа
                if len(results) > 1:
                    final_prompt = f"""User question: {query}

Tool results:
{tool_response}

Provide a friendly response. Respond with ONLY JSON: {{"response": "your answer"}}"""
                    final_response = llm_client(final_prompt)
                    # Очищаем от markdown
                    clean = final_response.strip()
                    if clean.startswith("```"):
                        clean = clean.strip("`").split("\n", 1)[1] if "\n" in clean else clean.strip("`")
                    try:
                        final_data = json.loads(clean)
                        final_answer = final_data.get("response", clean)
                    except:
                        final_answer = clean
                else:
                    final_answer = tool_response

                print(f"\n{'=' * 40}")
                print(f"Result: {final_answer}")
                print(f"{'=' * 40}\n")

            except json.JSONDecodeError:
                # LLM вернул текст напрямую
                clean_response = response.strip()
                if clean_response.startswith("```"):
                    clean_response = clean_response.strip("`").split("\n", 1)[1] if "\n" in clean_response else clean_response.strip("`")
                print(f"\n{'=' * 40}")
                print(f"Result: {clean_response}")
                print(f"{'=' * 40}\n")

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
#        "What is the time in Bengaluru?",  # Запрос времени
#        "What is the weather like right now in Dubai?",  # Запрос погоды
        "Сколько сейчас времени в Риге и какая погода в Риге?"
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