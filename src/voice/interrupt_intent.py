"""Fast, conservative text checks shared by both voice interruption paths."""

import re
import unicodedata


_CHINESE_BACKCHANNEL = re.compile(
    r"(?:嗯哼|嗯|啊|哦|噢|呃|对(?:的|啊|呀)?|是(?:的|啊|呀)?|好(?:的|啊|呀|吧|嘞)?|行)+"
)
_ENGLISH_BACKCHANNELS = frozenset(
    {
        "ok", "okay", "yes", "yeah", "yep", "yup", "right", "sure",
        "uhhuh", "hmm", "hmmm", "mm", "mmm", "isee", "gotit",
    }
)


def is_backchannel_text(text: str) -> bool:
    """Recognize a whole acknowledgement, never a prefix of a real request.

    The caller must separately check that a reply is in progress: the same
    words can be valid answers while the assistant is listening. Question
    marks remain significant, so an asking-back "对？" is not discarded.
    """
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    compact = "".join(
        char
        for char in normalized
        if not char.isspace()
        and (char == "?" or not unicodedata.category(char).startswith("P"))
    )
    if not compact or len(compact) > 24:
        return False
    return bool(
        _CHINESE_BACKCHANNEL.fullmatch(compact)
        or compact in _ENGLISH_BACKCHANNELS
    )
