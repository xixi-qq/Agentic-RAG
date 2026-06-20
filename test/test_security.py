import pytest

from utils.security import hash_password, verify_password


def test_hash_and_verify_password():
    password = "test123456"

    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(password, hashed) is True
    assert verify_password("wrong-password", hashed) is False


def test_same_password_generates_different_hashes():
    first = hash_password("test123456")
    second = hash_password("test123456")

    assert first != second
    assert verify_password("test123456", first) is True
    assert verify_password("test123456", second) is True


def test_password_over_72_bytes_is_rejected():
    password = "密" * 25

    with pytest.raises(ValueError, match="72 字节"):
        hash_password(password)


def test_invalid_hash_returns_false():
    assert verify_password("password", "invalid-hash") is False
