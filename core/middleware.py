from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from utils.jwt import decode_token


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self,request: Request,call_next):
        if request.url.path in [
            '/user/login/username',
            '/user/login/phone',
            '/user/register'
        ]:
            return await call_next(request)

        token = request.headers.get('Authorization')
        if not token:
            return JSONResponse(status_code=401,content={'message':'未提供token'})
        if not token.startswith("Bearer "):
            return JSONResponse(status_code=401, content={"message": "token格式错误"})

        token = token.removeprefix("Bearer ").strip()
        payload = decode_token(token)
        if not payload:
            return JSONResponse(status_code=401,content={'message':'token无效'})
        request.state.user = payload
        return await call_next(request)
