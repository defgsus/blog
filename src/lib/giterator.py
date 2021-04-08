import re
import sys
import tarfile
import datetime
import subprocess
from io import BytesIO
from typing import Generator, List, Tuple, Optional, Sequence

import dateutil.parser

_date_parser = dateutil.parser


def parse_datetime(s: str) -> datetime.datetime:
    return _date_parser.parse(s)


class Giterator:

    RE_CHANGE_NUMSTATS = re.compile(r"^(-|\d+)\s(-|\d+)\s(.*)$")
    RE_CHANGE_SUMMARY = re.compile(r"^([a-z]+) mode (\d\d\d\d\d\d) (.+)")
    MAX_TOKEN_LENGTH = 64
    LOG_INFOS = [
        ("%H", "hash"),
        ("%T", "tree_hash"),
        ("%P", "parent_hash", lambda s: s.split() if s.strip() else []),
        ("%an", "author"),
        ("%ae", "author_email"),
        ("%aI", "author_date", parse_datetime),
        ("%an", "committer"),
        ("%ae", "committer_email"),
        ("%aI", "committer_date", parse_datetime),
        ("%D", "ref_names", lambda s: s.split(", ") if s.strip() else []),
        ("%e", "encoding"),
    ]
    # something that should never appear in a git message
    DELIMITER1 = "$$$1-giterator-data-delimiter-$$$"
    DELIMITER2 = "\n$$$2-giterator-data-delimiter-$$$"

    def __init__(
            self,
            path: str,
            git_args: List[str] = None,
            verbose: bool = False,
    ):
        self.verbose = verbose
        self.path = path
        self._git_args = git_args or [
            "--all",
            "--reverse",
        ]
        self._num_commits = None
        self._hashes = set()

    def _log(self, *args):
        if self.verbose:
            print(*args, file=sys.stderr)

    def num_commits(self) -> int:
        if self._num_commits is None:
            self._log(" ".join(["git", "rev-list", "--count"] + self._git_args))
            output = subprocess.check_output(
                ["git", "rev-list", "--count"] + self._git_args,
                cwd=self.path,
                )
            self._num_commits = int(output)

        return self._num_commits

    def iter_commits(
            self,
            offset: int = 0,
            count: int = 0,
            json_compatible: bool = False,
    ) -> Generator[dict, None, None]:
        """
        Yields a dictionary for every git log that is found
        in the given directory.

        The ``git log`` command is used to get all the logs.

        :param offset: int
            Skip these number of commits before yielding.
            (via git log --skip parameter)

        :param count: int
            If > 0 then stop after this number of commits.

        :param json_compatible: bool
            If True, convert datetimes to strings

        :return: generator of dict
        """
        git_cmd = [
                      "git", "log",
                      "--numstat", "--summary",
                      f"--pretty={self.DELIMITER1}%n"
                      f"{'%n'.join(i[0] for i in self.LOG_INFOS)}"
                      f"%n%B{self.DELIMITER2}",
                  ] + self._git_args

        self._log(" ".join(git_cmd))
        process = subprocess.Popen(
            git_cmd,
            stdout=subprocess.PIPE,
            cwd=self.path
        )

        try:
            commit = dict()
            current_line = 0
            cur_count = 0
            while count <= 0 or (cur_count - offset) < count:
                line = process.stdout.readline()
                if not line:
                    break

                line = decode(line, ignore_errors=True).rstrip()

                # a new commit starts
                if line == self.DELIMITER1:
                    if commit:
                        if cur_count >= offset:
                            yield commit
                        cur_count += 1
                    commit = dict()
                    current_line = 0

                # commit message ended and changes (numstats) follow
                elif line == self.DELIMITER2[1:]:
                    commit["message"] = commit["message"].rstrip()
                    current_line = -1

                # digest each line
                else:
                    if 1 <= current_line <= len(self.LOG_INFOS):
                        log_info: Tuple = self.LOG_INFOS[current_line - 1]
                        value = line
                        if len(log_info) > 2:
                            value = log_info[2](value)
                        if json_compatible and isinstance(value, datetime.datetime):
                            value = value.isoformat()
                        commit[log_info[1]] = value

                    elif current_line == len(self.LOG_INFOS) + 1:
                        commit["message"] = line.rstrip()
                    elif current_line > len(self.LOG_INFOS) + 1:
                        commit["message"] += "\n" + line.rstrip()

                    elif current_line == -1:
                        line = line.strip()
                        if not self._parse_changes(commit, line):
                            self._parse_summary(commit, line)

                if current_line >= 0:
                    current_line += 1

            if commit:
                if cur_count >= offset and (count <= 0 or (cur_count - offset) < count):
                    yield commit

        finally:
            process.kill()
            process.wait()

    def iter_files(
            self,
            treeish: str,
            filenames: Optional[Sequence[str]] = None
    ) -> Generator[Tuple[BytesIO, tarfile.TarInfo], None, None]:
        """
        Iterates through all files at a commit or tree,
        by reading the tar output of `git archive`.

        :param treeish: str
        :param filenames: optional list of paths or filenames
        :return: generator of File
        """
        git_cmd = [
            "git", "archive", "--format=tar", treeish,
        ]
        if filenames:
            git_cmd += list(filenames)

        self._log(" ".join(git_cmd))
        process = subprocess.Popen(
            git_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.path
        )

        try:
            # TODO: would be nice if we could iterate through
            #   the files in the stdout stream but
            #   tarfile lib requires seekable streams
            #   so the whole tar-file has to be in memory

            tar_data = BytesIO(process.stdout.read())
            error_response = process.stderr.read()

        finally:
            process.kill()
            process.wait()

        tar_data.seek(0)

        try:
            for tarinfo in tarfile.open(fileobj=tar_data):
                if tarinfo.isfile():
                    yield File(self, tar_data, tarinfo)

        except tarfile.ReadError:
            if not filenames:
                return
            raise tarfile.ReadError(error_response.decode("utf-8"))

    def _parse_changes(self, commit: dict, line: str) -> bool:
        change_match = self.RE_CHANGE_NUMSTATS.match(line)
        if not change_match:
            return False

        if "changes" not in commit:
            commit["changes"] = []

        additions, deletions, name = change_match.groups()

        # TODO: additions/deletions should be integer converted
        #   but might be "-" in case of binary files
        commit["changes"].append({
            "name": name,
            "type": "change",
            "additions": additions,
            "deletions": deletions,
        })

        rename = get_git_renaming(name)
        if rename:
            commit["changes"][-1].update({
                "name": rename[1],
                "old_name": rename[0],
                "type": "rename"
            })

    def _parse_summary(self, commit: dict, line: str) -> bool:
        if line.startswith("rename "):
            return True

        change_match = self.RE_CHANGE_SUMMARY.match(line.strip())
        if not change_match:
            return False

        type, mode, name = change_match.groups()

        #if commit.get("changes"):
        for ch in commit["changes"]:
            if ch["name"] == name:
                ch["type"] = type
                ch["mode"] = mode
                return True

        #if not commit.get("changes"):
        #    commit["changes"] = []
        #    commit["changes"].append
        #print(commit)
        #print(type, mode, name)

        raise AssertionError(
            f"Expected '{name}' in --netstat changes, but got only --summary '{line}'\ncommit: {commit}"
        )


def decode(s: bytes, ignore_errors: bool = False) -> Optional[str]:
    for encoding in ("latin1", "utf-8"):
        try:
            return s.decode(encoding)
        except UnicodeDecodeError:
            pass
    if ignore_errors:
        return s.decode("utf-8", errors="ignore")
    else:
        return None


import tarfile
from typing import Dict
from io import BytesIO


class File:
    """
    Representation of a file, kept in memory, by reading a `git archive` stream.
    """

    def __init__(self, repo, buffer: BytesIO, tarinfo: tarfile.TarInfo):
        self.repo = repo
        self._buffer = buffer
        self._tarinfo = tarinfo
        self._data = None

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}', {self.size}, {self.mtime})"

    @property
    def name(self) -> str:
        return self._tarinfo.name

    @property
    def size(self) -> int:
        return self._tarinfo.size

    @property
    def mtime(self) -> int:
        return self._tarinfo.mtime

    @property
    def data(self) -> bytes:
        if self._data is None:
            if self._tarinfo.sparse is not None:
                self._data = b""
                for offset, size in self._tarinfo.sparse:
                    self._buffer.seek(offset)
                    self._data += self._buffer.read(size)
            else:
                self._buffer.seek(self._tarinfo.offset_data)
                self._data = self._buffer.read(self._tarinfo.size)

        return self._data

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "size": self.size,
            "mtime": self.mtime,
            "data": self.data,
        }

    def byte_histogram(self) -> Dict[int, int]:
        counter = dict()
        for b in self.data:
            counter[b] = counter.get(b, 0) + 1

        return counter


RE_MULTI_SLASH = re.compile(r"/+")


def get_git_renaming(name: str) -> Optional[Tuple[str, str]]:
    if " => " not in name:
        return

    try:
        idx_start = name.index("{")
        idx_end = name.index("}")
    except ValueError:
        name1, name2 = name.split(" => ")
        return name1, name2

    middle = name[idx_start+1:idx_end].split(" => ")
    name1 = name[:idx_start] + middle[0] + name[idx_end+1:]
    name2 = name[:idx_start] + middle[1] + name[idx_end+1:]
    return RE_MULTI_SLASH.sub("/", name1), RE_MULTI_SLASH.sub("/", name2)
