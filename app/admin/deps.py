from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.admin.service import AdminService
from app.chat.deps import ChatRepositoryDep


def get_admin_service(repository: ChatRepositoryDep) -> AdminService:
    return AdminService(repository)


AdminServiceDep = Annotated[AdminService, Depends(get_admin_service)]
