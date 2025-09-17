from core.rag.splitter.fixed_text_splitter import FixedRecursiveCharacterTextSplitter


def test_fixed_splitter_groups_short_lines_until_chunk_size() -> None:
    splitter = FixedRecursiveCharacterTextSplitter(
        chunk_size=25,
        chunk_overlap=0,
        fixed_separator="\n",
    )

    text = "Pain!\nSo painful!\nMy head hurts so much!"

    chunks = splitter.split_text(text)

    assert chunks == ["Pain!\nSo painful!", "My head hurts so much!"]


def test_fixed_splitter_preserves_consecutive_separators() -> None:
    splitter = FixedRecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
        fixed_separator="\n",
    )

    text = "A\n\nB"

    chunks = splitter.split_text(text)

    assert chunks == ["A\n\nB"]
