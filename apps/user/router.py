from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import JSONResponse

from apps.user.crud import get_user_by_username, get_user_by_phone, create_user
from config.db_config import get_db
from apps.user.schemas import UserRegister, PhoneLogin
from utils.jwt import create_access_token
from utils.security import verify_password

router = APIRouter(prefix='/user',tags=['user'])

@router.post('/register')
async def register(user_data:UserRegister,db: AsyncSession = Depends(get_db)):
    existing_name = await get_user_by_username(user_data.name,db)
    if existing_name:
        raise HTTPException(status_code=400, detail="用户名已存在")
    existing_phone = await get_user_by_phone(user_data.phone,db)
    if existing_phone:
        raise HTTPException(status_code=400, detail="该手机号已注册")
    user = await create_user(user_data,db)
    response_data = {
        'id':user.id,
        'name':user.name,
        'phone':user.phone,
        'gender':user.gender
    }
    return JSONResponse(status_code=201,content=response_data)


def check_login_password(user, password: str):
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    correct = verify_password(password, user.password)
    if not correct:
        raise HTTPException(status_code=401, detail="用户名或密码错误")


def build_login_response(user):
    token = create_access_token(
        data={
            "user_id": user.id,
            "username": user.name,
        })
    response_data = {
        'access_token': token,
        'token_type':"bearer",
        'user_info':{
            'id':user.id,
            'name':user.name,
            'phone':user.phone,
            'gender':user.gender
        }
    }
    return JSONResponse(status_code=201,content=response_data)


@router.post('/login/username')
async def login_by_username(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_username(form_data.username, db)
    check_login_password(user, form_data.password)
    return build_login_response(user)


@router.post('/login/phone')
async def login_by_phone(
    user_data: PhoneLogin,
    db: AsyncSession = Depends(get_db)
):
    user = await get_user_by_phone(user_data.phone, db)
    check_login_password(user, user_data.password)
    return build_login_response(user)

