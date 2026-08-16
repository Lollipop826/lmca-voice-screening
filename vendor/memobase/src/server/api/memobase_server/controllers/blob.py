import pydantic
from sqlalchemy.exc import IntegrityError

from ..models.utils import Promise
from ..models.database import BlobIdempotencyKey, BufferZone, GeneralBlob
from ..models.response import CODE, BlobData, IdData
from ..models.blob import ChatBlob, DocBlob, BlobType
from ..connectors import Session
from ..utils import get_blob_token_size


def _source_turn_id(blob: ChatBlob | DocBlob) -> str:
    value = (blob.fields or {}).get("source_turn_id")
    return str(value).strip() if value is not None else ""


async def insert_blob(
    user_id: str,
    project_id: str,
    blob: BlobData,
    *,
    enqueue_buffer: bool = False,
) -> Promise[IdData]:
    try:
        blob_parsed = blob.to_blob()
    except pydantic.ValidationError as e:
        return Promise.reject(CODE.BAD_REQUEST, f"Unable to parse blob: {e}")
    source_turn_id = _source_turn_id(blob_parsed)
    with Session() as session:
        if source_turn_id:
            existing = (
                session.query(BlobIdempotencyKey)
                .filter_by(
                    user_id=user_id,
                    project_id=project_id,
                    source_turn_id=source_turn_id,
                )
                .one_or_none()
            )
            if existing:
                return Promise.resolve(IdData(id=existing.blob_id, duplicate=True))
        blob_db = GeneralBlob(
            blob_type=blob_parsed.type,
            blob_data=blob_parsed.get_blob_data(),
            additional_fields=blob_parsed.fields,
            user_id=user_id,
            project_id=project_id,
        )
        try:
            session.add(blob_db)
            session.flush()
            if source_turn_id:
                session.add(
                    BlobIdempotencyKey(
                        user_id=user_id,
                        source_turn_id=source_turn_id,
                        blob_id=blob_db.id,
                        project_id=project_id,
                    )
                )
            if enqueue_buffer:
                session.add(
                    BufferZone(
                        user_id=user_id,
                        blob_id=blob_db.id,
                        blob_type=blob_parsed.type,
                        token_size=get_blob_token_size(blob_parsed),
                        project_id=project_id,
                    )
                )
            session.commit()
        except IntegrityError:
            session.rollback()
            if source_turn_id:
                existing = (
                    session.query(BlobIdempotencyKey)
                    .filter_by(
                        user_id=user_id,
                        project_id=project_id,
                        source_turn_id=source_turn_id,
                    )
                    .one_or_none()
                )
                if existing:
                    return Promise.resolve(
                        IdData(id=existing.blob_id, duplicate=True)
                    )
            raise
        return Promise.resolve(IdData(id=blob_db.id))


async def get_blob(user_id: str, project_id: str, blob_id: str) -> Promise[BlobData]:
    with Session() as session:
        blob_db = (
            session.query(GeneralBlob)
            .filter_by(id=blob_id, user_id=user_id, project_id=project_id)
            .one_or_none()
        )
        if not blob_db:
            return Promise.reject(
                CODE.NOT_FOUND, f"Blob with id {blob_id} of user {user_id} not found"
            )
        rt_blob = BlobData(
            blob_type=BlobType(blob_db.blob_type),
            blob_data=blob_db.blob_data,
            fields=blob_db.additional_fields,
            created_at=blob_db.created_at,
            updated_at=blob_db.updated_at,
        )
        return Promise.resolve(rt_blob)


async def remove_blob(user_id: str, project_id: str, blob_id: str) -> Promise[None]:
    with Session() as session:
        blob_db = (
            session.query(GeneralBlob)
            .filter_by(id=blob_id, user_id=user_id, project_id=project_id)
            .one_or_none()
        )
        if not blob_db:
            return Promise.resolve(None)
        else:
            session.delete(blob_db)
            session.commit()
    return Promise.resolve(None)
