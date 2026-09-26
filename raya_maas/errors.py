class APIError(Exception):
    def __init__(self, message: str, status: int = 400, code: str = "invalid_request_error"):
        self.message = message
        self.status = status
        self.code = code
        super().__init__(message)

    def body(self) -> dict:
        kind = "server_error" if self.status >= 500 else self.code
        return {"error": {"message": self.message, "type": kind, "param": None, "code": self.code}}
