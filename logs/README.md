FOLDER : logs

Purpose :

Structured application logs written by `logger.py` (see root
README's Logging section).

Layout :

logs/<YYYY-MM-DD>/algo.log

A new dated subfolder is created automatically each day. Each
`algo.log` is a rotating file (5 MB per file, 5 backups kept) shared
by every module that calls `logger.get_logger(__name__)`.

Note :

This folder is safe to clear/delete - logs regenerate automatically
on the next run. Consider adding `logs/` to `.gitignore` if it isn't
already, since log contents can grow and may include runtime data.
