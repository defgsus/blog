---
layout: post
title: The unsafe CLIP image search
custom_js: 
  - clip/clip-image-search.js
---

Welcome seeker!

This is an example of [OpenAI](https://openai.com/)'s 
[CLIP](https://arxiv.org/pdf/2103.00020.pdf) model used for image search on *random web images*.

CLIP is a new and astonishing neural network that is able to transform images and text
into the same latent space. To quote wikipedia, 
[latent variables](https://en.wikipedia.org/wiki/Latent_variable) 
are *variables that are not directly observed but are rather inferred 
through a mathematical model*. In this case, CLIP transforms any image or text into a list
of 512 numbers. These can be used to measure similarity between images and/or texts.

The model was trained on 400 million image-text pairs scraped from the
internet (until end of 2019). I recommend to read OpenAI's blog posts about 
[CLIP](https://openai.com/blog/clip/)
and [multimodal neurons](https://openai.com/blog/multimodal-neurons/) 
(and [here](https://distill.pub/2021/multimodal-neurons/)) and frankly any other
publication of that amazing team.

The trained CLIP-model has been released by OpenAI as a 
[python torch package](https://github.com/openai/CLIP) 
and can freely be exploited for a couple of things. The most obvious is searching for images
using natural language. 

Since this is a static blog page, i'm just providing a a fixed set of 
search prompts for each of which the top 5 images have been selected via CLIP. 

It's much more impressive to use a free text prompt and this demo does not do the CLIP model 
much justice but it's a start. This is **not** searching among all the images of the 
internet. Only a handful. Technical details are below the search form. 


<div class="image-search">
    
    <hr>
    <div class="labels"></div>    
    <label style="color: #999;">
        <input type="checkbox" class="cb-show-images"> 
        <b>load images</b> (That's an opt-in to actually load images from all over the web)
    </label>
    <hr>

    <br>
    
    <div class="results">
    </div>
</div>
<hr>


### tech details

The images have been collected in early 2021 by browsing the top 10k websites according
to the [Tranco Top Sites](https://tranco-list.eu) list, plus some extra. Details can be 
found in [this post]({% post_url 2021-03-22-top-10k-referrers %}). 

About **180k images** could be collected from the HAR recordings. 
They contain a lot of porn, hence the title of this post.

This is just a fun demo and no in-depth research review but i'd like to scratch a
topic. Since CLIP was trained on *the internet* it might have seen some of the images
in this post together with their textual counterparts. It's probably quite hard to actually
test a model's performance on specific tasks if it's not exactly clear what has been in 
the training data. 

These new *internet-fed models* are actually called **pre-trained** to 
distinguish them from models **trained on a specific task**. To accurately measure 
performance, models are tested against a **test set**, not the **training set**. 
Researchers are aware of this and try to filter out parts of test sets that might have been 
sucked into the pre-training data. Personally, i think the believability of accuracy of the old 
*training set vs. test set* methodology is challenged by those new models.

For programming details i peeked a bit at 
[ShivamShrirao/CLIP_Image_Search](https://github.com/ShivamShrirao/CLIP_Image_Search)
and [haltakov/natural-language-image-search](https://github.com/haltakov/natural-language-image-search).

When playing with the search it was obvious that CLIP prefers certain images, 
more or less independently of certain search prompts. For example 
`a photo of a naked woman` and `a photo of a naked man` have the same top image.
I think it's possible to fine-tune the image search be collecting the similarities
for a wide range of orthogonal search prompts per image and then rescale the 
similarity score.

Determining the image features took about 75 minutes on a GTX 1660, 
with a batch size of 50 images. The text features of the 7000 fixed prompts only 
took a few minutes. 

The final json file with the top-5 urls per search prompt is 5Mb large. I actually wanted to
add a lot more things and adjectives and show longer result lists but the size of 
this json file grows exponentially. 

Please be aware, that the image urls are completely copied from the HAR archive so they 
might contain tracking-specific query parameters and all sorts of weird stuff. A couple 
of images will be blocked by extensions like **Privacy Badger** or **uBlock Origin**.

### moral details

CLIP has watched a lot of porn. This can be determined by how well it works to identify
images using *porn language*. It's possible to lower the relevance of images matching `dick` 
if you don't like them and raise the relevance for `green hair and pink panties` if that's
what you want.

And if one compares the type of images it returns for **woman**
and **man**, i'd say CLIP is definitely a man. Or is it that there are not enough 
image-text pairs in the internet where all the little details of a photo of a naked
man are described?

Generally, one can assume that the social biases that are present in our society
are learned and reproduced by such internet-bred models. They are fun to use and
those biases might even help certain people finding their relevant stuff but in
general i'm scared by the imagination that jurisdiction, law enforcement and
whatever else is using these models as decision makers or helpers at least.

I guess it's fair to say that a model trained on the internet is actually trained on
a pile of shit. There's some good stuff in there but i do not assume that my kids
would become *good people* by just playing with their smart-phones. 

Fortunately, CLIP has been trained before the whole corona fuck-up, so it's still 
associating the word corona with bright yellow streams of light. So happy that
this snapshot has been saved...
 
  