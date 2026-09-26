from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class APIError(Exception):
    def __init__(self, message: str, status: int = 422, code: str = "invalid_request_error"):
        self.message = message
        self.status = status
        self.code = code
        super().__init__(message)

    def body(self) -> dict:
        return {"error": {"message": self.message, "code": self.code}}
