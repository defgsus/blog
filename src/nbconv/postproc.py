from nbconvert.postprocessors import PostProcessorBase


class JekyllPostProcessor(PostProcessorBase):

    def postprocess(self, input):
        # 'input' is just a path
        # print("POSTPROC", type(input), input)
        return input
