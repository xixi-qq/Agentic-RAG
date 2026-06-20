import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
import dotenv

dotenv.load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token


def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login/username")


def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    :param token:
    :return: payload={
            "user_id": user.id,
            "username": user.name,
            "exp": expire
        }
    """
    payload = decode_token(token)

    if not payload:
        raise HTTPException(status_code=401, detail="token无效")

    return payload
