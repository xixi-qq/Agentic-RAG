from typing import Optional

from pydantic import BaseModel, Field, model_validator


class UserRegister(BaseModel):
    name: str
    password: str = Field(max_length=72)
    repassword: str = Field(None, alias="rePassword")
    phone: str = Field(None,pattern=r"^1[3-9]\d{9}$",description='手机号')

    @model_validator(mode="after")
    def check_passwords(self):
        if self.password != self.repassword:
            raise ValueError("两次密码不一致")
        return self

class PhoneLogin(BaseModel):
    phone: str = Field(pattern=r"^1[3-9]\d{9}$", description="手机号")
    password: str


class Userinfo(BaseModel):
    id: int
    name: str
    phone: str
    gender: Optional[str]
