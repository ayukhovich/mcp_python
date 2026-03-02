"""
Главный файл MCP сервера с поддержкой SSE
Запускает сервер и регистрирует инструменты
"""

import sys

from colorama import Fore, Style  # Для цветного вывода в консоль
from mcp.server.fastmcp import FastMCP

from config import HOST, PORT, LOG_DIR, LOG_FILE, LOG_LEVEL, USE_MOCK, TRANSPORT
from logging_config import setup_logging
from mcp_handler import MCPHandler

# Настройка логирования
logger = setup_logging(LOG_FILE, LOG_LEVEL)

# Показываем конфигурацию
logger.info("=" * 60)
logger.info(f"📋 CONFIGURATION")
logger.info("=" * 60)
logger.info(f"📦 TRANSPORT: {TRANSPORT}")
logger.info(f"🌐 HOST: {HOST}")
logger.info(f"🔌 PORT: {PORT}")
logger.info(f"🎭 USE_MOCK: {USE_MOCK}")
logger.info(f"📁 Log directory: {LOG_DIR}")
logger.info(f"📝 Log file: {LOG_FILE}")
logger.info("=" * 60)

# Создаем обработчик инструментов
handler = MCPHandler(logger)

mcp = FastMCP(
    name="WeatherTimeServer",
    stateless_http=True,
    json_response=True,
    debug=True,
    log_level="DEBUG"
)


# --- РЕГИСТРАЦИЯ ИНСТРУМЕНТОВ ---
@mcp.tool()
def time_tool(timezone: str = None) -> str:
    "Provides the current time for a given city's timezone like Asia/Kolkata, America/New_York etc. If no timezone is provided, it returns the local time."

    logger.info(f"{Fore.CYAN}⚡ Executing time_tool{Style.RESET_ALL}")
    result = handler.time_tool(timezone)
    logger.info(f"{Fore.GREEN}✓ time_tool completed{Style.RESET_ALL}")
    return result


@mcp.tool()
def weather_tool(location: str) -> str:
    """Provides weather information for a given location"""

    logger.info(f"{Fore.CYAN}⚡ Executing weather_tool{Style.RESET_ALL}")
    result = handler.weather_tool(location)
    logger.info(f"{Fore.GREEN}✓ weather_tool completed{Style.RESET_ALL}")
    return result


# --- ЗАПУСК СЕРВЕРА ---
if __name__ == "__main__":
    try:
        logger.info(f"🚀 Starting MCP Server on {HOST}:{PORT}")
        logger.info(f"📋 Available tools: time_tool, weather_tool")
        logger.info(f"🔗 SSE endpoint: http://{HOST}:{PORT}/sse")
        logger.info(f"📡 Message endpoint: http://{HOST}:{PORT}/messages/")
        logger.info("=" * 60)

        # Запускаем сервер
        mcp.run(
            transport=TRANSPORT
        )
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        sys.exit(1)
