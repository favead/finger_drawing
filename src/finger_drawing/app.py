"""Запуск приложения"""


class App:
    """Хранилище основных компонентов системы,
    реализация логики приложения"""

    def __init__(self) -> None:
        pass

    def run(self) -> None:
        print("run")
        return None


def run() -> None:
    app = App()
    app.run()
    return None
