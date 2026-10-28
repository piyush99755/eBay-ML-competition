def convert_tags_to_bio(tags: list[str]) -> list[str]:
    """
    Convert eBay competition tags into BIO-style labels.

    Rules:
    - A normal non-empty aspect tag starts a new entity -> B-<tag>
    - An empty tag continues the previous entity
    - O and continuations of O remain O
    """

    bio_tags = []
    previous_entity_tag = None

    for tag in tags:

        if tag == "O":
            bio_tags.append("O")

            # Remember that the current entity is O in case
            # the next token has an empty continuation tag.
            previous_entity_tag = "O"

        elif tag == "":
            if previous_entity_tag is None:
                raise ValueError(
                    "Found a continuation tag without a previous entity."
                )

            if previous_entity_tag == "O":
                bio_tags.append("O")
            else:
                bio_tags.append(f"I-{previous_entity_tag}")

        else:
            bio_tags.append(f"B-{tag}")
            previous_entity_tag = tag

    return bio_tags