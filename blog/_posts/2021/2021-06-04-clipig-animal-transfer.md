---
layout: post
title: CLIPig animal transfer 
custom_js: clip/clipig-animals.js 
custom_css: clipig-animals.css
---

Once the astonishment and the feeling of complete personal incompetence 
has diminished after reading the amazing article about 
[DALL-E](https://openai.com/dall-e???), i gathered some courage back and
investigated the methods that OpenAI used to create those images.

DALL-E is based on the [GPT3](???) transformer. The [paper](???) is 
overwhelmingly complex, not only in terms of the algorithms but also 
with regard to reducing and distributing computation across hundreds of
expensive GPUs to allow training convergence within *only* weeks 
instead of years.

The major components of GPT3 and DALL-E are not publicly released so
there's no way for an average middle-class individual to reproduce 
those results. The [CLIP](https://github.com/openai/CLIP/)
network on the other hand, which was used to pick the best images
produced by DALL-E, **is** available to the middle class.

So i started playing with it a couple of weeks ago which resulted in 
the [CLIP Image Generator](https://github.com/defgsus/clipig/), 
a little framework for creating images out of the blue, using CLIP
as an art critique.

Below is a reproduction of the *animal concept transfer* experiment
in the DALL-E article using only the CLIP network. You can use the 
**up** and **down** cursor keys to select a thing and the **1** and
**2** key respectively to select the rendering method.

<div class="clipig-input"></div>


## Details

I chose the example because i liked the *penguins made of piano*
so much. Some of them are really artistic in the sense that: 
If a human artist selects one of the top-ranked images it becomes
true art.

When one interacts with the prompt selector in the article, 
the list of keywords and corresponding text prompts is 
printed to the web-console so it's easy to reuse. 


### DALL-E vs. CLIPig

DALL-E creates images by completing a *sentence*, similar to GPT3, 
where each token describes an 8² pixel window. A sentence in natural
language and optionally some initial image tokens are provided and
DALL-E generates the next image tokens by the best of it's knowledge.

In CLIPig, an image (starting with noise) is constantly updated to 
increase the similarity between it's feature and a target feature,
usually created via a natural language text prompt. 
The features are calculated by CLIP and the optimization of the
feature loss, coupled with a few hand-designed transformations 
during each training step, may eventually lead to an image that a
human does indeed recognize as the depiction of what the text prompt
suggests.
 

### Computation time and number of images

Each image shows CLIPig's **first try**. There is no ranking among 512
different runs as in the DALL-E article because i simply can not 
afford the computation time. Rendering a single image took
**137 seconds for method 1** and **194 seconds for method 2**. There are
9 animals and 90 different *things* which resulted in a rendering
time of **31** and **44 hours** respectively. Multiplying that with 512
leads to several years of computation time. At 50 Watts/h. Which i 
don't think reasonable at any rate.

I guess that generating an image with DALL-E only takes a few seconds
as it is a completely different method.

Some experiments with ranking multiple runs where carried out. 
They are shown [below](#Ranking-experiment). But first the two
methods will be briefly described.


### Method 1

A 224² pixel image is initialized with a bilinear-scaled 10² pixel noise. 
Some random rotation and a little noise is applied before each 
CLIP evaluation. During the first 10% of training, the resulting image 
is blurred after each training step. Below is the complete 
[CLIPig configuration](https://defgsus.github.io/clipig/) script.

```yaml
epochs: 200
optimizer: rmsprob
learnrate: 1.0

init:
  mean: 0.33
  std: 0.03
  resolution: 10

targets:
  - batch_size: 10
    features:
      - text: a cat made of salami. a cat with the texture of a salami.
    transforms:
      - pad:
          size: 32
          mode: edge
      - random_rotate:
          degree: -30 30
          center: 0 1
      - mul: 1/2
      - mean: .5
      - noise: 1/10

postproc:
- blur:
    kernel_size: 3
  end: 10%
```


### Method 2

The major difference to method 1 is the resolution. In the beginning
of the training, a 8² pixel image is optimized. The resolution increases
to 16², 32², 64², 128² and finally 224² pixels during the first half
of the training. The idea behind this is that optimizing a very low
resolution image will encourage CLIP to create *proper* objects in 
the first place without increasing the feature similarity through
fine (adversarial) details. Once a nice object is rendered CLIP may
work out the details.

As can be seen by comparing the images of the two methods, this 
resolution trick leads to a very different style.  

The text prompt was shortened to the first sentence, *'an animal made of thing.'*
and the second sentence *'an animal with the texture of a thing.'* was dropped
because i subjectively liked most of the resulting images better. 

Here's the configuration:

```yaml
epochs: 300
resolution: 
  - '{0: 8, 1: 8, 2: 16, 3: 32, 4: 64, 5: 128}.get(int(t*16), 224)' 
optimizer: rmsprob
learnrate: 1.0

init:
  mean: 0.3
  std: .01

targets:
  - batch_size: 10
    features:
      - text: a cat made of salami.
    transforms:
      - resize: 224
      - pad:
          size: 32
          mode: edge
      - random_rotate:
          degree: -20 20
          center: 0 1
      - random_crop: 224
      - mul: 1/3
      - mean: 0.5
      - bwnoise: 1/5*ti
    constraints:
      - blur: 
          kernel_size: 11
          start: 50%
      - saturation:
          below: 0.005
```


### Ranking experiment

Images in CLIPig emerge by progressively *massaging* 
pixel planes into something that CLIP would rate as a high match to 
the target feature. The whole process is stochastic and each
run produces a different image with a slightly different 
feature similarity.

In the following experiment two of the example text prompts have been
rendered 512 times each. The image training was reduced to 20 epochs,
instead of 200 or 300. The best 6 matches, according to CLIP, where selected
and rendered for another 300 epochs.

Method 1 starts with 10x10 random pixels, so it's already pointing CLIP
to a certain space within the similarity should be increased. 

Method 2 starts by generating a small (8x8, then 16x16) image to help 
CLIP concentrate on the proper position of objects before working on 
the details. In this case, the best 16x16 images where selected for
further rendering. I did not see a lot of difference between 
many of those small images and stopped the initial runs at some point.

The following table shows each prompt's (or method's) 6 best
images and 2 worst, rated after the initial run. The **sim** numbers
show the rating results after the initial run on the left and
after the final run on the right.

<div class="clipig-ranking">
<table>
    <tr>
        <td><cite>snail of harp</cite> (method 1)</td>
        <td><cite>penguin of piano</cite> (method 1)</td>
        <td><cite>penguin of piano</cite> (method 2)</td>
    </tr>
    <tr>
        <td>best of 512 (<a href="../../../assets/data/clipig/snail-of-harp.csv">csv</a>)</td>
        <td>best of 512 (<a href="../../../assets/data/clipig/penguin-of-piano.csv">csv</a>)</td>
        <td>best of 149 (<a href="../../../assets/data/clipig/penguin-of-piano-reso.csv">csv</a>)</td>
    </tr>

<tr>
<td>
            <div id="img-snail-of-harp-487" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-487.png">
                <div class="desc">
                    <div class="left">#487</div> 
                    <div class="right">sim: <b>36.81</b> ⭢ <b>45.49</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-344" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-344.png">
                <div class="desc">
                    <div class="left">#344</div> 
                    <div class="right">sim: <b>35.30</b> ⭢ <b>48.65</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-108" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-108.png">
                <div class="desc">
                    <div class="left">#108</div> 
                    <div class="right">sim: <b>21.86</b> ⭢ <b>46.65</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-134" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-134.png">
                <div class="desc">
                    <div class="left">#134</div> 
                    <div class="right">sim: <b>36.77</b> ⭢ <b>45.77</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-43" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-43.png">
                <div class="desc">
                    <div class="left">#43</div> 
                    <div class="right">sim: <b>35.23</b> ⭢ <b>47.54</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-80" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-80.png">
                <div class="desc">
                    <div class="left">#80</div> 
                    <div class="right">sim: <b>21.56</b> ⭢ <b>47.80</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-326" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-326.png">
                <div class="desc">
                    <div class="left">#326</div> 
                    <div class="right">sim: <b>36.60</b> ⭢ <b>45.48</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-260" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-260.png">
                <div class="desc">
                    <div class="left">#260</div> 
                    <div class="right">sim: <b>34.38</b> ⭢ <b>48.01</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-137" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-137.png">
                <div class="desc">
                    <div class="left">#137</div> 
                    <div class="right">sim: <b>21.39</b> ⭢ <b>49.67</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-268" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-268.png">
                <div class="desc">
                    <div class="left">#268</div> 
                    <div class="right">sim: <b>36.50</b> ⭢ <b>45.07</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-40" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-40.png">
                <div class="desc">
                    <div class="left">#40</div> 
                    <div class="right">sim: <b>34.11</b> ⭢ <b>49.50</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-42" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-42.png">
                <div class="desc">
                    <div class="left">#42</div> 
                    <div class="right">sim: <b>21.29</b> ⭢ <b>50.49</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-471" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-471.png">
                <div class="desc">
                    <div class="left">#471</div> 
                    <div class="right">sim: <b>36.45</b> ⭢ <b>45.77</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-221" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-221.png">
                <div class="desc">
                    <div class="left">#221</div> 
                    <div class="right">sim: <b>33.00</b> ⭢ <b>49.65</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-21" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-21.png">
                <div class="desc">
                    <div class="left">#21</div> 
                    <div class="right">sim: <b>21.28</b> ⭢ <b>49.92</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-38" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-38.png">
                <div class="desc">
                    <div class="left">#38</div> 
                    <div class="right">sim: <b>36.35</b> ⭢ <b>45.61</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-244" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-244.png">
                <div class="desc">
                    <div class="left">#244</div> 
                    <div class="right">sim: <b>31.69</b> ⭢ <b>38.75</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-24" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-24.png">
                <div class="desc">
                    <div class="left">#24</div> 
                    <div class="right">sim: <b>21.24</b> ⭢ <b>47.43</b></div>
                </div>
            </div>
        </td>
</tr>

    <tr><td><h4>worst of 512</h4></td> <td><h4>worst of 512</h4></td> <td><h4>worst of 149</h4></td></tr>                    
    
<tr>
<td>
            <div id="img-snail-of-harp-214" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-214.png">
                <div class="desc">
                    <div class="left">#214</div> 
                    <div class="right">sim: <b>28.86</b> ⭢ <b>45.38</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-57" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-57.png">
                <div class="desc">
                    <div class="left">#57</div> 
                    <div class="right">sim: <b>20.22</b> ⭢ <b>46.67</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-6" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-6.png">
                <div class="desc">
                    <div class="left">#6</div> 
                    <div class="right">sim: <b>17.80</b> ⭢ <b>48.58</b></div>
                </div>
            </div>
        </td>
</tr>
<tr>
<td>
            <div id="img-snail-of-harp-171" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/snail-of-harp-from-171.png">
                <div class="desc">
                    <div class="left">#171</div> 
                    <div class="right">sim: <b>29.40</b> ⭢ <b>46.18</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-390" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-390.png">
                <div class="desc">
                    <div class="left">#390</div> 
                    <div class="right">sim: <b>20.82</b> ⭢ <b>47.87</b></div>
                </div>
            </div>
        </td>
<td>
            <div id="img-penguin-of-piano-106" class="image-wrapper">
                <img src="../../../assets/images/clipig/dall-e-samples/penguin-of-piano-from-106.png">
                <div class="desc">
                    <div class="left">#106</div> 
                    <div class="right">sim: <b>18.48</b> ⭢ <b>48.61</b></div>
                </div>
            </div>
        </td>
</tr>


</table>
</div>


## Conclusion

The CLIPig images are not nearly as good as the DALL-E samples, 
both in terms of quality and *freshness* of the composition idea.

The ranking experiment shows that it's quite hard to create
convincing shapes and forms just from noisy training on pixel
planes. 

Looking at the deviation of the initial and final 
ranking values, we can see that a couple of hundred epochs
of training in CLIPig will raise the similarity value to
high levels in almost any case, no matter the initial
state of the image. The one exception being 
[penguin #244](#img-penguin-of-piano-244)
where the penguin is actually nicely coalescing with the
piano but the fine boundaries and details could not be worked
out.
