---
layout: post
title: Malazan CLIP features 
---

This is starting to consume a serious amount of my work time, not to 
mention the rest of the day. I need to talk about it. 

![CLIP generated visual]({../../../../../assets/images/clip/malazan6.png)

This is what drives CLIP's image encoder to match with the features of 
the text encoder, given the word *Malazan*. You can read about OpenAI's
CLIP network and feature visualization in general all over the web and
i'm too lazy to put links right now, except for images.

CLIP's image encoder has an input window of 224x224 pixels. To render
high resolution images the input window is placed randomly all over the place.

![CLIP generated visual]({../../../../../assets/images/clip/malazan6-rot.png)

In the image above, the **rotation** is also randomized a lot. Obviously, CLIP associates
*Malazan* with fantasy book covers. Maybe even specific ones. From the 
rotation it gets inspired to draw those magical mandala things. They might be
from any fantasy book but i think it's from the Malazan series. Although CLIP
never spells it right. Let's say, it's enough for fan. 

The rotation also seems to make those soldiers more of the fighting sort.  

And the minutes pass by and i think, just one more picture! Then i should really
do something else. And more minutes pass by. 

So now i can fill billions of pixels with the malazan pattern in some way
but i find it desirable to let CLIP also control the whole composition, not
only little input windows. These images are 1024² pixels so CLIP's 224² input
window really just sees a small section at once. If we scale down the image 
randomly before feeding to CLIP, it looks like this:

![CLIP generated visual]({../../../../../assets/images/clip/malazan6-scale.png)

The section with the people at the bottom looks really cool. It's much larger 
than the CLIP input so the scaling seems to help. Think of it as resizing a small
resolution image and then fine-tuning the details. I'm completely amazed since
a couple of days now by what is possible. 

And another ten minutes later...

![CLIP generated visual]({../../../../../assets/images/clip/malazan6-perspective.png)

Here, each input window has been transformed by a random perspective projection. 
Note that not so much *Malazan* has been tried to write down. And it failed even
more where it was tried. The perspective book pages are also kind of nice.

There is probably a way to get rid of the text. It's a quite interesting topic too.
Via *linear probing* of the output features tied to some classic image recognition 
data-set, one can find determine specific weight surfaces for text, texture and 
possibly a lot of other things. Once determined, those weights can be used to 
raise or lower the training loss during image rendering.

Let's keep the random zoom feature and add some variation. For example, we can test a 
chosen CLIP window for multiple features at the same time and apply the one 
that matches best.

Trying **Malazan** and **Lord of the Rings**:

![CLIP generated visual]({../../../../../assets/images/clip/malazan7-lotr.png)

Okay, it tried to smirch it into the book title but apart there is not much difference.
This also highlights the fact that the *Malazan Book of the Fallen* is much more
complex and absorbing than *Lord of the Rings*. 

Let's try a few other things:

![CLIP generated visual]({../../../../../assets/images/clip/malazan9-cthulhu.png)

Here are the number of training frames that each text feature received:

```
malazan               : 879 (29.46%)
sword                 : 665 (22.29%)
cthulhu               : 435 (14.58%)
metal armour          : 575 (19.27%)
stairs in a dungeon   : 217  (7.27%)
leather armour        : 144  (4.83%)
warrior               :  69  (2.31%)
```

Amazing! These 10 minutes where really worth the time! *Cthulhu* seems to be known
well to the CLIP. I'll add the perspective randomization to the *Malazan* target
to remove the titles a bit and rerun the experiment. 

![CLIP generated visual]({../../../../../assets/images/clip/malazan10-cthulhu.png)

Some rendering artifacts have triggered the *stairs in a dungeon* and they got
a large amount of representation.  

Here's a Cthulhu-only image:
  
![CLIP generated visual]({../../../../../assets/images/clip/cthulhu1.png)

There's a guy with a suit in there? Must be *Bob Howard* from *The Laundry Files*?

Back to the idea of composition.

In the following images, those detail training steps
are combined with training a resized CLIP window that fits the whole image. 
Of course this is quite blurry but we can turn it off at some point 
during training and let the detail training handle the rest.

Here's a full-scale training step for *Malazan landscape*, by which CLIP does not seem 
to understand *views of nature* but rather characteristic fantasy book maps. 
    
![CLIP generated visual]({../../../../../assets/images/clip/malazan-landscape-training.png)

So the quality is terrible and there are a lot of artifacts and glitches that come
from either the convolutional network layers in CLIP or from the methods trying to 
minimize those artifacts. Mainly random translation and gaussian blurring.  

But the final image, just some 7 minutes later!

![CLIP generated visual]({../../../../../assets/images/clip/malazan-landscape.png)

It has some global structure as well as a lot of tiny details. It's not completely
rational but that's okay with my fantasy image generation desires. Note that it did
not find good detailed matches for the mountains and the bigger letters. Also there's
a bit too much swords in there and a bit too much crayon drawing for my taste. 

Here i kept *Malazan landscape* for the global composition but only allowed *Malazan*
for the details:

![CLIP generated visual]({../../../../../assets/images/clip/malazan-landscape3-training.png)

![CLIP generated visual]({../../../../../assets/images/clip/malazan-landscape3.png)

I really need to figure out how to remove text if i want to go further in the Malazan
universe!

Here's an example from a different universe where the big-blurred composition vs. 
uncomposed details worked quite well. It's a combination of 
*a photo of two androids walking through a futuristic city* and stuff like 
*close-up of a sad robot*, *close-up of various machinery parts* and so on.

![CLIP generated visual]({../../../../../assets/images/clip/androids5.png)

I will explore more. There's a couple of technical and artistic problems that
i want to fix. But, you know, it takes time.. 
