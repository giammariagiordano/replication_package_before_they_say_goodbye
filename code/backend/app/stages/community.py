from sqlalchemy import select

from ..community import accommodation, citizenship
from ..models import Alias, Comment, Developer, DeveloperMonth, Issue
from ..pipeline import Stage, StageContext
from .text import clean_text


class CommunityStage(Stage):
    name = "community"
    label = "Community bond"
    description = (
        "Citizenship (messages on other people's and newcomers' issues/PRs) and linguistic accommodation "
        "(cross-entropy under the language model of the other developers' messages)."
    )

    def run(self, ctx: StageContext):
        pid = ctx.project.id
        dev_of = dict(ctx.db.execute(select(Alias.id, Alias.developer_id).where(Alias.project_id == pid)).all())
        bots = set(ctx.db.scalars(select(Developer.id).where(Developer.project_id == pid, Developer.is_bot)))
        first = dict(ctx.db.execute(select(Developer.id, Developer.first_activity).where(Developer.project_id == pid)).all())

        def human(alias_id):
            dev = dev_of.get(alias_id)
            return None if dev is None or dev in bots else dev

        threads, authored, texts = {}, [], []
        for number, alias_id, created, body in ctx.db.execute(
            select(Issue.number, Issue.alias_id, Issue.created_at, Issue.body).where(Issue.project_id == pid)
        ):
            if (dev := human(alias_id)) is not None:
                threads[number] = (dev, created)
                authored.append((dev, created, number))
                texts.append((dev, created, clean_text(body)))
        for number, alias_id, created, body in ctx.db.execute(
            select(Comment.issue_number, Comment.alias_id, Comment.created_at, Comment.body).where(Comment.project_id == pid)
        ):
            if (dev := human(alias_id)) is not None:
                authored.append((dev, created, number))
                texts.append((dev, created, clean_text(body)))

        helping = citizenship(threads, authored, first)
        ctx.progress(1, 2, "Language models")
        language = accommodation(texts)
        rows = ctx.db.scalars(select(DeveloperMonth).where(DeveloperMonth.project_id == pid)).all()
        for r in rows:
            c = helping.get((r.developer_id, r.month), {})
            r.thread_messages = c.get("thread_messages", 0)
            r.others_messages = c.get("others_messages", 0)
            r.newcomer_messages = c.get("newcomer_messages", 0)
            r.xent_messages, r.xent_sum = language.get((r.developer_id, r.month), (0, 0.0))
        ctx.progress(2, 2, f"{len(language)} developer-months with accommodation")
        ctx.log(f"Community bond: {len(helping)} developer-months with conversations, {len(language)} with accommodation")
