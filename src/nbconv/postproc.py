import os
import sys

from nbconvert.postprocessors import PostProcessorBase


def printe(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class JekyllPostProcessor(PostProcessorBase):

    def postprocess(self, input):
        print("INPZ", input)
        with open(input) as fp:
            text = fp.read()

        # move references to assets
        filename = ".".join(os.path.basename(input).split(".")[:-1]) + "_files"

        changed_text = text.replace(f"{filename}/", f"{{{{site.baseurl}}}}/assets/nb/{filename}/")

        if changed_text != text:
            printe(f"Postprocessing {input}")
            with open(input, "w") as fp:
                fp.write(changed_text)

        return input
