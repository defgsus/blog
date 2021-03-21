---
layout: post
custom_css: 
  - jquery.dataTables.min.css
custom_js: 
  - jquery-3.6.0.min.js
  - jquery.dataTables-1.10.24.min.js
  - plotly.min.js
  - require-stub.js
title: ad servers today
enable: plotly datatables
tags: web ads privacy

---


Is this serious? 

I actually did not realize how boggy the internet has become until, after years, i turned off the script blockers out of curiosity and visited a random mainstream website. It was rather disturbing. It takes quite some effort to dive into this topic, and the extent and complexity of interconnected ad and profilig networks is simply crazy and probably out of control.

### Just an example 

Here's the roundup of about **90 newspaper and magazine websites**. They have been chosen using public (ad-related) databases with the restriction to have at least 50,000 printed copies per issue and, of course, a website. They needed to be matched by name across two databases and this is what came out. Some websites have been added by hand.  ([source code](https://github.com/defgsus/blog/tree/master/src/har_research/german_papers) and [notebook](https://github.com/defgsus/blog/blob/master/src/har_research/german_papers/ad-servers-today.ipynb))  

On each site, [Alice](https://defgsus.github.io/blog/2021/03/05/alice-website.html)

- clicked the *yes-accept-all-just-leave-me-alone* button
- scrolled the page down to the bottom
- clicked one article and scrolled about

The following table is conveniently ordered by **third parties**, but can be ordered in any way you like if javascript is enabled. Third-party means anything that got requested by the browser which is from a different domain name than the actual website. To be honest, i did not distinguish between first-party [CDN](https://en.wikipedia.org/wiki/Content_delivery_network) and true third-party requests because it would require manual examination and it was tiring enough to browse through all the pages in the first place.

(If you think one should write a script that reliably clicks the accept-all button on all pages that have one, you're welcome to join efforts. For some reason it's made very hard)





<table border="1" class="dataframe compact" id="table-4940c80a57faeff3231f">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>shareholders</th>
      <th>third parties</th>
      <th>requests</th>
      <th>tp-requests</th>
      <th>tp-requests %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>www.saarbruecker-zeitung.de</th>
      <td>Saarbrücker Zeitung</td>
      <td>33</td>
      <td>107</td>
      <td>1438</td>
      <td>1281</td>
      <td>89.08</td>
    </tr>
    <tr>
      <th>www.waz.de</th>
      <td>Westdeutsche Allgemeine Zeitung</td>
      <td>10</td>
      <td>106</td>
      <td>1456</td>
      <td>1272</td>
      <td>87.36</td>
    </tr>
    <tr>
      <th>www.general-anzeiger-bonn.de</th>
      <td>General-Anzeiger</td>
      <td>32</td>
      <td>105</td>
      <td>1326</td>
      <td>1325</td>
      <td>99.92</td>
    </tr>
    <tr>
      <th>www.volksfreund.de</th>
      <td>Trierischer Volksfreund</td>
      <td>34</td>
      <td>96</td>
      <td>1359</td>
      <td>1157</td>
      <td>85.14</td>
    </tr>
    <tr>
      <th>www.aachener-nachrichten.de</th>
      <td>Aachener Nachrichten</td>
      <td>49</td>
      <td>90</td>
      <td>958</td>
      <td>754</td>
      <td>78.71</td>
    </tr>
    <tr>
      <th>www.abendzeitung-muenchen.de</th>
      <td>Abendzeitung</td>
      <td>6</td>
      <td>89</td>
      <td>1947</td>
      <td>1785</td>
      <td>91.68</td>
    </tr>
    <tr>
      <th>www.merkur.de</th>
      <td>Dachauer Nachrichten</td>
      <td>26</td>
      <td>85</td>
      <td>1415</td>
      <td>1336</td>
      <td>94.42</td>
    </tr>
    <tr>
      <th>www.swp.de</th>
      <td>Alb-Bote</td>
      <td>29</td>
      <td>83</td>
      <td>987</td>
      <td>881</td>
      <td>89.26</td>
    </tr>
    <tr>
      <th>www.welt.de</th>
      <td>Die Welt</td>
      <td>18</td>
      <td>82</td>
      <td>1238</td>
      <td>1010</td>
      <td>81.58</td>
    </tr>
    <tr>
      <th>www.kreiszeitung.de</th>
      <td>Kreiszeitung</td>
      <td>38</td>
      <td>82</td>
      <td>1045</td>
      <td>984</td>
      <td>94.16</td>
    </tr>
    <tr>
      <th>www.all-in.de</th>
      <td>Allgäuer Zeitung</td>
      <td>6</td>
      <td>82</td>
      <td>919</td>
      <td>787</td>
      <td>85.64</td>
    </tr>
    <tr>
      <th>www.nordbayern.de</th>
      <td>Altmühl-Bote</td>
      <td>14</td>
      <td>80</td>
      <td>1180</td>
      <td>962</td>
      <td>81.53</td>
    </tr>
    <tr>
      <th>www.echo-online.de</th>
      <td>Darmstädter Echo</td>
      <td>41</td>
      <td>79</td>
      <td>863</td>
      <td>783</td>
      <td>90.73</td>
    </tr>
    <tr>
      <th>www.tz.de</th>
      <td>Amberger Zeitung</td>
      <td>27</td>
      <td>79</td>
      <td>1245</td>
      <td>1156</td>
      <td>92.85</td>
    </tr>
    <tr>
      <th>www.augsburger-allgemeine.de</th>
      <td>Augsburger Allgemeine</td>
      <td>3</td>
      <td>77</td>
      <td>1072</td>
      <td>853</td>
      <td>79.57</td>
    </tr>
    <tr>
      <th>www.handelsblatt.com</th>
      <td>Handelsblatt</td>
      <td>11</td>
      <td>75</td>
      <td>1012</td>
      <td>641</td>
      <td>63.34</td>
    </tr>
    <tr>
      <th>www.nwzonline.de</th>
      <td>Nordwest-Zeitung</td>
      <td>9</td>
      <td>75</td>
      <td>1252</td>
      <td>986</td>
      <td>78.75</td>
    </tr>
    <tr>
      <th>www.suedkurier.de</th>
      <td>Südkurier</td>
      <td>5</td>
      <td>75</td>
      <td>841</td>
      <td>681</td>
      <td>80.98</td>
    </tr>
    <tr>
      <th>www.wiesbadener-kurier.de</th>
      <td>Idsteiner Zeitung</td>
      <td>38</td>
      <td>74</td>
      <td>923</td>
      <td>851</td>
      <td>92.20</td>
    </tr>
    <tr>
      <th>www.bz-berlin.de</th>
      <td>B.Z.</td>
      <td>18</td>
      <td>72</td>
      <td>1254</td>
      <td>1117</td>
      <td>89.07</td>
    </tr>
    <tr>
      <th>www.fr.de</th>
      <td>Frankfurter Rundschau</td>
      <td>28</td>
      <td>72</td>
      <td>1147</td>
      <td>1038</td>
      <td>90.50</td>
    </tr>
    <tr>
      <th>spiegel.de</th>
      <td>Der Tagesspiegel</td>
      <td>12</td>
      <td>72</td>
      <td>1076</td>
      <td>759</td>
      <td>70.54</td>
    </tr>
    <tr>
      <th>www.azonline.de</th>
      <td>Allgemeine Zeitung</td>
      <td>22</td>
      <td>71</td>
      <td>1000</td>
      <td>905</td>
      <td>90.50</td>
    </tr>
    <tr>
      <th>www.main-echo.de</th>
      <td>Bote vom Unter-Main</td>
      <td>7</td>
      <td>71</td>
      <td>1197</td>
      <td>1140</td>
      <td>95.24</td>
    </tr>
    <tr>
      <th>www.wz.de</th>
      <td>WZ Westdeutsche Zeitung</td>
      <td>33</td>
      <td>70</td>
      <td>739</td>
      <td>604</td>
      <td>81.73</td>
    </tr>
    <tr>
      <th>www.freiepresse.de</th>
      <td>Freie Presse</td>
      <td>18</td>
      <td>70</td>
      <td>537</td>
      <td>418</td>
      <td>77.84</td>
    </tr>
    <tr>
      <th>www.lvz.de</th>
      <td>Döbelner Allgemeine</td>
      <td>43</td>
      <td>70</td>
      <td>1064</td>
      <td>991</td>
      <td>93.14</td>
    </tr>
    <tr>
      <th>www.rnz.de</th>
      <td>Nordbadische Nachrichten</td>
      <td>5</td>
      <td>69</td>
      <td>997</td>
      <td>810</td>
      <td>81.24</td>
    </tr>
    <tr>
      <th>www.mz-web.de</th>
      <td>Mitteldeutsche Zeitung</td>
      <td>39</td>
      <td>69</td>
      <td>481</td>
      <td>348</td>
      <td>72.35</td>
    </tr>
    <tr>
      <th>www.moz.de</th>
      <td>Gransee-Zeitung</td>
      <td>29</td>
      <td>69</td>
      <td>755</td>
      <td>645</td>
      <td>85.43</td>
    </tr>
    <tr>
      <th>www.westfalen-blatt.de</th>
      <td>Bünder Tageblatt</td>
      <td>18</td>
      <td>69</td>
      <td>1084</td>
      <td>937</td>
      <td>86.44</td>
    </tr>
    <tr>
      <th>www.donaukurier.de</th>
      <td>Donaukurier</td>
      <td>7</td>
      <td>67</td>
      <td>929</td>
      <td>747</td>
      <td>80.41</td>
    </tr>
    <tr>
      <th>www.stuttgarter-zeitung.de</th>
      <td>Stuttgarter Zeitung</td>
      <td>60</td>
      <td>67</td>
      <td>871</td>
      <td>694</td>
      <td>79.68</td>
    </tr>
    <tr>
      <th>www.ln-online.de</th>
      <td>Bad Schwartauer Nachrichten</td>
      <td>32</td>
      <td>67</td>
      <td>1005</td>
      <td>962</td>
      <td>95.72</td>
    </tr>
    <tr>
      <th>www.focus.de</th>
      <td>Focus</td>
      <td>10</td>
      <td>67</td>
      <td>1087</td>
      <td>802</td>
      <td>73.78</td>
    </tr>
    <tr>
      <th>www.stuttgarter-nachrichten.de</th>
      <td>Fellbacher Zeitung</td>
      <td>17</td>
      <td>66</td>
      <td>816</td>
      <td>619</td>
      <td>75.86</td>
    </tr>
    <tr>
      <th>www.lr-online.de</th>
      <td>Lausitzer Rundschau</td>
      <td>7</td>
      <td>66</td>
      <td>721</td>
      <td>595</td>
      <td>82.52</td>
    </tr>
    <tr>
      <th>www.noz.de</th>
      <td>Bersenbrücker Kreisblatt</td>
      <td>10</td>
      <td>66</td>
      <td>789</td>
      <td>737</td>
      <td>93.41</td>
    </tr>
    <tr>
      <th>www.volksstimme.de</th>
      <td>Volksstimme</td>
      <td>6</td>
      <td>64</td>
      <td>1064</td>
      <td>984</td>
      <td>92.48</td>
    </tr>
    <tr>
      <th>www.ovb-online.de</th>
      <td>Chiemgau-Zeitung</td>
      <td>26</td>
      <td>64</td>
      <td>781</td>
      <td>694</td>
      <td>88.86</td>
    </tr>
    <tr>
      <th>www.ostsee-zeitung.de</th>
      <td>Ostsee-Zeitung</td>
      <td>35</td>
      <td>64</td>
      <td>936</td>
      <td>885</td>
      <td>94.55</td>
    </tr>
    <tr>
      <th>www.schwarzwaelder-bote.de</th>
      <td>Schwarzwälder Bote</td>
      <td>60</td>
      <td>64</td>
      <td>946</td>
      <td>725</td>
      <td>76.64</td>
    </tr>
    <tr>
      <th>www.neuepresse.de</th>
      <td>Neue Presse</td>
      <td>29</td>
      <td>63</td>
      <td>1113</td>
      <td>1050</td>
      <td>94.34</td>
    </tr>
    <tr>
      <th>www.stimme.de</th>
      <td>Heilbronner Stimme</td>
      <td>17</td>
      <td>63</td>
      <td>857</td>
      <td>768</td>
      <td>89.61</td>
    </tr>
    <tr>
      <th>www.goettinger-tageblatt.de</th>
      <td>Eichsfelder Tageblatt</td>
      <td>31</td>
      <td>62</td>
      <td>801</td>
      <td>746</td>
      <td>93.13</td>
    </tr>
    <tr>
      <th>www.onetz.de</th>
      <td>Amberger Zeitung</td>
      <td>13</td>
      <td>61</td>
      <td>834</td>
      <td>593</td>
      <td>71.10</td>
    </tr>
    <tr>
      <th>www.bild.de</th>
      <td>BILD-Zeitung</td>
      <td>18</td>
      <td>61</td>
      <td>1319</td>
      <td>937</td>
      <td>71.04</td>
    </tr>
    <tr>
      <th>www.mainpost.de</th>
      <td>Bote vom Haßgau</td>
      <td>6</td>
      <td>60</td>
      <td>694</td>
      <td>461</td>
      <td>66.43</td>
    </tr>
    <tr>
      <th>www.wa.de</th>
      <td>Allgemeine Laber-Zeitung</td>
      <td>27</td>
      <td>60</td>
      <td>823</td>
      <td>751</td>
      <td>91.25</td>
    </tr>
    <tr>
      <th>www.freitag.de</th>
      <td>der Freitag</td>
      <td>4</td>
      <td>59</td>
      <td>682</td>
      <td>537</td>
      <td>78.74</td>
    </tr>
    <tr>
      <th>www.tagesspiegel.de</th>
      <td>Der Tagesspiegel</td>
      <td>12</td>
      <td>59</td>
      <td>847</td>
      <td>639</td>
      <td>75.44</td>
    </tr>
    <tr>
      <th>www.svz.de</th>
      <td>Der Prignitzer</td>
      <td>16</td>
      <td>56</td>
      <td>1375</td>
      <td>1081</td>
      <td>78.62</td>
    </tr>
    <tr>
      <th>www.rp-online.de</th>
      <td>Bergische Morgenpost</td>
      <td>36</td>
      <td>56</td>
      <td>499</td>
      <td>330</td>
      <td>66.13</td>
    </tr>
    <tr>
      <th>www.kn-online.de</th>
      <td>Kieler Nachrichten</td>
      <td>30</td>
      <td>56</td>
      <td>909</td>
      <td>840</td>
      <td>92.41</td>
    </tr>
    <tr>
      <th>www.nordkurier.de</th>
      <td>Nordkurier</td>
      <td>35</td>
      <td>56</td>
      <td>520</td>
      <td>432</td>
      <td>83.08</td>
    </tr>
    <tr>
      <th>www.muensterschezeitung.de</th>
      <td>Grevener Zeitung</td>
      <td>19</td>
      <td>55</td>
      <td>813</td>
      <td>659</td>
      <td>81.06</td>
    </tr>
    <tr>
      <th>www.maz-online.de</th>
      <td>Brandenburger Kurier</td>
      <td>42</td>
      <td>54</td>
      <td>945</td>
      <td>868</td>
      <td>91.85</td>
    </tr>
    <tr>
      <th>www.heise.de</th>
      <td>Heise Online</td>
      <td>4</td>
      <td>54</td>
      <td>832</td>
      <td>691</td>
      <td>83.05</td>
    </tr>
    <tr>
      <th>www.faz.net</th>
      <td>Frankfurter Allgemeine Sonntagszeitung</td>
      <td>12</td>
      <td>52</td>
      <td>692</td>
      <td>409</td>
      <td>59.10</td>
    </tr>
    <tr>
      <th>www.rheinpfalz.de</th>
      <td>Die Rheinpfalz</td>
      <td>17</td>
      <td>50</td>
      <td>682</td>
      <td>511</td>
      <td>74.93</td>
    </tr>
    <tr>
      <th>www.badische-zeitung.de</th>
      <td>Badische Zeitung</td>
      <td>12</td>
      <td>49</td>
      <td>648</td>
      <td>470</td>
      <td>72.53</td>
    </tr>
    <tr>
      <th>www.infranken.de</th>
      <td>Bayerische Rundschau</td>
      <td>25</td>
      <td>46</td>
      <td>515</td>
      <td>417</td>
      <td>80.97</td>
    </tr>
    <tr>
      <th>www.zeit.de</th>
      <td>Die Zeit</td>
      <td>16</td>
      <td>45</td>
      <td>719</td>
      <td>452</td>
      <td>62.87</td>
    </tr>
    <tr>
      <th>www.morgenweb.de</th>
      <td>Bergsträßer Anzeiger</td>
      <td>26</td>
      <td>44</td>
      <td>628</td>
      <td>625</td>
      <td>99.52</td>
    </tr>
    <tr>
      <th>www.berliner-kurier.de</th>
      <td>Berliner Kurier</td>
      <td>4</td>
      <td>41</td>
      <td>329</td>
      <td>303</td>
      <td>92.10</td>
    </tr>
    <tr>
      <th>www.nw.de</th>
      <td>Bad Oeynhausener Kurier</td>
      <td>27</td>
      <td>39</td>
      <td>634</td>
      <td>528</td>
      <td>83.28</td>
    </tr>
    <tr>
      <th>www.aerztezeitung.de</th>
      <td>Ärzte Zeitung</td>
      <td>15</td>
      <td>37</td>
      <td>328</td>
      <td>240</td>
      <td>73.17</td>
    </tr>
    <tr>
      <th>www.schwaebische.de</th>
      <td>Aalener Nachrichten</td>
      <td>40</td>
      <td>35</td>
      <td>464</td>
      <td>349</td>
      <td>75.22</td>
    </tr>
    <tr>
      <th>www.sueddeutsche.de</th>
      <td>Süddeutsche Zeitung</td>
      <td>43</td>
      <td>34</td>
      <td>547</td>
      <td>232</td>
      <td>42.41</td>
    </tr>
    <tr>
      <th>www.saechsische.de</th>
      <td>Chemnitzer Morgenpost</td>
      <td>12</td>
      <td>28</td>
      <td>374</td>
      <td>238</td>
      <td>63.64</td>
    </tr>
    <tr>
      <th>taz.de</th>
      <td>die tageszeitung</td>
      <td>2</td>
      <td>27</td>
      <td>1106</td>
      <td>839</td>
      <td>75.86</td>
    </tr>
    <tr>
      <th>www.berliner-zeitung.de</th>
      <td>Berliner Zeitung</td>
      <td>4</td>
      <td>26</td>
      <td>236</td>
      <td>188</td>
      <td>79.66</td>
    </tr>
    <tr>
      <th>bnn.de</th>
      <td>Acher- und Bühler Bote</td>
      <td>2</td>
      <td>23</td>
      <td>269</td>
      <td>208</td>
      <td>77.32</td>
    </tr>
    <tr>
      <th>www.epochtimes.de</th>
      <td>The Epoch Times</td>
      <td>2</td>
      <td>23</td>
      <td>372</td>
      <td>238</td>
      <td>63.98</td>
    </tr>
    <tr>
      <th>deutsche-wirtschafts-nachrichten.de</th>
      <td>Deutsche Wirtschaftsnachrichten</td>
      <td>5</td>
      <td>22</td>
      <td>350</td>
      <td>285</td>
      <td>81.43</td>
    </tr>
    <tr>
      <th>www.rhein-zeitung.de</th>
      <td>Nahe-Zeitung</td>
      <td>5</td>
      <td>21</td>
      <td>311</td>
      <td>191</td>
      <td>61.41</td>
    </tr>
    <tr>
      <th>www.architekturzeitung.com</th>
      <td>Architekturzeitung</td>
      <td>0</td>
      <td>20</td>
      <td>797</td>
      <td>615</td>
      <td>77.16</td>
    </tr>
    <tr>
      <th>www.idowa.de</th>
      <td>Allgemeine Laber-Zeitung</td>
      <td>3</td>
      <td>20</td>
      <td>278</td>
      <td>147</td>
      <td>52.88</td>
    </tr>
    <tr>
      <th>jungefreiheit.de</th>
      <td>Junge Freiheit</td>
      <td>5</td>
      <td>20</td>
      <td>509</td>
      <td>275</td>
      <td>54.03</td>
    </tr>
    <tr>
      <th>www.mittelbayerische.de</th>
      <td>Bayerwald Echo</td>
      <td>11</td>
      <td>18</td>
      <td>295</td>
      <td>150</td>
      <td>50.85</td>
    </tr>
    <tr>
      <th>linkezeitung.de</th>
      <td>Linke Zeitung</td>
      <td>0</td>
      <td>16</td>
      <td>273</td>
      <td>270</td>
      <td>98.90</td>
    </tr>
    <tr>
      <th>www.kath.net</th>
      <td>Katholische Nachrichten</td>
      <td>0</td>
      <td>15</td>
      <td>158</td>
      <td>64</td>
      <td>40.51</td>
    </tr>
    <tr>
      <th>www.medical-tribune.de</th>
      <td>Medical Tribune</td>
      <td>0</td>
      <td>14</td>
      <td>124</td>
      <td>84</td>
      <td>67.74</td>
    </tr>
    <tr>
      <th>www.sonntagsblatt.de</th>
      <td>Sonntagsblatt</td>
      <td>1</td>
      <td>13</td>
      <td>314</td>
      <td>211</td>
      <td>67.20</td>
    </tr>
    <tr>
      <th>www.generalanzeiger.de</th>
      <td>Generalanzeiger</td>
      <td>32</td>
      <td>12</td>
      <td>240</td>
      <td>133</td>
      <td>55.42</td>
    </tr>
    <tr>
      <th>www.die-tagespost.de</th>
      <td>Die Tagespost</td>
      <td>2</td>
      <td>12</td>
      <td>203</td>
      <td>84</td>
      <td>41.38</td>
    </tr>
    <tr>
      <th>www.boersen-zeitung.de</th>
      <td>Börsenzeitung</td>
      <td>0</td>
      <td>10</td>
      <td>201</td>
      <td>41</td>
      <td>20.40</td>
    </tr>
    <tr>
      <th>www.bayernkurier.de</th>
      <td>Bayernkurier</td>
      <td>0</td>
      <td>5</td>
      <td>88</td>
      <td>12</td>
      <td>13.64</td>
    </tr>
    <tr>
      <th>www.neues-deutschland.de</th>
      <td>Neues Deutschland</td>
      <td>6</td>
      <td>4</td>
      <td>138</td>
      <td>15</td>
      <td>10.87</td>
    </tr>
    <tr>
      <th>www.juedische-allgemeine.de</th>
      <td>Jüdische Allgemeine</td>
      <td>1</td>
      <td>2</td>
      <td>175</td>
      <td>15</td>
      <td>8.57</td>
    </tr>
    <tr>
      <th>www.das-parlament.de</th>
      <td>Das Parlament</td>
      <td>1</td>
      <td>1</td>
      <td>56</td>
      <td>4</td>
      <td>7.14</td>
    </tr>
    <tr>
      <th>www.jungewelt.de</th>
      <td>Junge Welt</td>
      <td>3</td>
      <td>0</td>
      <td>85</td>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table><script type="text/javascript">
        jQuery("#table-4940c80a57faeff3231f").DataTable({"order": [[3, "desc"]], "paging": true});
    </script><div class="table-description"><ul><li><b>shareholders</b>: Number of shareholders with at least 1% of capital share</li><li><b>third parties</b>: Number of networks requested that are not the main server</li><li><b>requests</b>: Number of network requests during the visit of the website</li><li><b>tp-requests</b>: Number of network requests to third-party servers during the visit of the website</li><li><b>tp-requests %</b>: Percent of network request to third-party servers compared to first-party servers</li></ul></div>


So imagine, there are people just using their browser as-is, reading an article in the *Saarbrücker Zeitung* and tell over a hundred other networks or companies what they are doing. 

What *exactly* happens there is subject of more research but the majority of those third-party requests can be attributed to **ads/header bidding** and **profiling**. You can type these words into the *major search engine* and it will gladly invite you with tutorials and documentation to setup these amazing technologies on your own page. My humble understanding at the moment is that whenever a browser displays one of *those* websites it screams out: 

> "I have these blank spaces in between my articles! Anybody any ad you wanna display? Let's say for a dollar per click?"

Eventually a third-party **ad exchange network** will deliver a piece of html that contains an ad. The website runner is called the **publisher**, the blank spaces are **inventory** and it's sold to **advertisers** which in turn work on behalf of the **brands**. And because it's all done algorithmically it's called **programmatic**.

For quite some time, the **waterfall** method was the widely used method for maximizing income of all participants in the advertising process. 

> "Does my trusted first-party ad supplier has an ad for this user? No? Well, how about my second-loveliest worldwide-top-ads supplier? A dollar, great!"

This method was argued to be a bit slow. Also, the publisher may miss ads for a higher price by not looking into the 5th remnant-crap-exchange. So it got gradually replaced by organizing auctions: 

> "Hey everyone! Anybody an ad? Floor price one dollar. Oh, you there would pay 1.20? Congratulations! Now gimme that ad."

And why would anyone pay those 1.20? Because it's a well measured prosperous website. Because an ad might perfectly fit to the content of the visited page or sub-page and therefore must generate a high level of **leads**. And because *something* in all the data bits transmitted from the browser to the advertisers tells them which ad might be most successful for the particular and potential customer using the browser. This is archived by the even more complicated and mysterious art of profiling and combination of several **first-party publisher data** with whatever else can be collected and attributed to a digital identity.

So here's already a good explanation for most of the thrird-party web traffic: The more, the better! 

Advertisers and exchangers are a bit concerned since a couple of years because publishers not only relied on the bidding process of exchange networks to deliver the best ads for the best price but they actually started to **waterfall the exchanges**: 

> "Hey, would someone please run an auction at 2 dollars? Nothing today? Well, i was just thinking, maybe someone had the perfect ad for this identity. 1.75 then? Great! I'll keep your ad for the moment and just run another auction to make sure.." 

And most of this happens in the browser, not on the publisher's server. In part you can even see the prices for the delivered ads when reading through the network log. 

The list of all third-party networks is [below](#third-parties-table).

### remark about shareholders

The **shareholders** column in above table shows the number of shareholders with at least 1% of capital share. This data is compiled by an institution called *Kommission zur Ermittlung der Konzentration im Medienbereich* ([KEK](https://www.kek-online.de/)). For example, you can [look up the ownership of Saarbrücker Zeitung](https://www.kek-online.de/medienkonzentration/mediendatenbank#/profile/media/5be1a6b1-2485-4efd-b291-d10211e901a4) on their interactive webpage. They also have an undocumented [API](https://medienvielfaltsmonitor.de/api/v1/media/) where the shareholder data was scraped from. There's only six websites/papers in my dataset which are not covered by the KEK data. They have zero shareholders in the table.

Here is a complete shareholder graph for the mentioned paper:




    (online) www.saarbruecker-zeitung.de
    └─Saarbrücker Zeitung Verlag und Druckerei GmbH
      ├─56.07 Rheinisch-Bergische Verlagsgesellschaft mbH
      │ ├─21.92 Befa Beteiligungs-GmbH
      │ │ ├─22.1 Merz-Betz, Florian
      │ │ ├─11.25 Merz, Thomas
      │ │ ├─11.25 Sader-Merz, Caroline
      │ │ ├─11.25 Merz, Esther
      │ │ ├─7.0 Stilz, Clara
      │ │ ├─7.0 Stilz, Leoni
      │ │ ├─6.0 Berger, Sarina
      │ │ ├─6.0 Berger, Viola
      │ │ ├─6.0 Berger, Thalita
      │ │ ├─5.35 Stilz, Markus
      │ │ ├─3.95 Stilz, Andreas
      │ │ ├─1.85 Stilz, Eva
      │ │ └─1.0 Betz, Esther
      │ ├─20.87 Wenderoth GmbH & Co KG
      │ │ ├─24.75 Ebel, Martin
      │ │ ├─24.75 Ebel, Stefan
      │ │ ├─24.75 Ebel, Johannes
      │ │ ├─24.75 Breitkreutz, Elisabeth
      │ │ ├─1.0 Wenderoth-Alt, Irene
      │ │ └─0.0 Wenderoth Verwaltungs-GmbH
      │ ├─9.9 Büro Dr. M. Droste GmbH & Co KG
      │ │ ├─89.6 Droste, Manfred
      │ │ ├─2.6 Droste, Tilman
      │ │ ├─2.6 Droste-Zobel, Lieselotte
      │ │ ├─2.6 Droste, Alexander
      │ │ ├─2.6 Droste, Felix
      │ │ └─0.0 Dr. M. Droste Verwaltungs GmbH
      │ ├─6.12 Girardet Verlag KG
      │ │ ├─35.8 Girardet, Klaus
      │ │ ├─28.0 Girardet, Rainer
      │ │ ├─12.2 Rheinische Post Verlagsgesellschaft mbH
      │ │ │ └─100.0 Rheinisch-Bergische Verlagsgesellschaft mbH
      │ │ │   └─...
      │ │ ├─8.2 Girardet, Nikolaus
      │ │ ├─8.2 Girardet, Isabelle
      │ │ ├─2.6 Girardet-Seiffert, Annette
      │ │ ├─2.6 Böhmer, Bärbel
      │ │ ├─2.6 Joens-Girardet, Christina
      │ │ ├─0.0 Girardet Verlag Verwaltungs GmbH
      │ │ └─0.0 Girardet, Dr. Michael
      │ ├─4.28 Lohse, Stephan
      │ ├─4.16 Klostermann, Thomas
      │ ├─4.16 Seifert, Katja
      │ ├─3.33 Droste, Tilman
      │ ├─3.33 Droste-Zobel, Lieselotte
      │ ├─3.33 Droste, Alexander
      │ ├─3.33 Droste, Felix
      │ ├─3.1 Arnold, Philipp
      │ ├─3.1 Arnold, Dr. Karl Hans
      │ ├─2.5 Lohse, Julia
      │ ├─2.5 Lohse, Benedikt
      │ ├─1.58 Rauert, Stephanie
      │ ├─0.97 Seifert, Felix
      │ ├─0.51 Rauert, Annabelle
      │ ├─0.51 Rauert, Konstantin
      │ └─0.51 Rauert, Robert
      ├─27.86 Gesellschaft für staatsbürgerliche Bildung Saar mbH
      │ ├─40.0 Union Stiftung e.V.
      │ ├─40.0 Demokratische Gesellschaft Saarland e. V.
      │ └─20.0 Villa Lessing Liberale Stiftung Saar e.V.
      └─16.07 Beteiligungsgesesellschaft Saarbrücker Zeitung GbR
        └─100.0 ca. 1.000 Mitarbeiter der Unternehmensgruppe Saarbrücker Zeitung


On first inspection i thought the number of shareholders is somehow related to the number of third-party servers but apart from the marketing outliers on the left side there does not seem to be a tight relation.









<div>                            <div id="dfe7857f-7b1b-40cd-8c48-07f3df391060" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("dfe7857f-7b1b-40cd-8c48-07f3df391060")) {                    Plotly.newPlot(                        "dfe7857f-7b1b-40cd-8c48-07f3df391060",                        [{"hovertemplate": "variable=shareholders<br>index=%{x}<br>value=%{y}<extra></extra>", "legendgroup": "shareholders", "line": {"color": "#636efa", "dash": "solid"}, "mode": "lines", "name": "shareholders", "orientation": "v", "showlegend": true, "type": "scatter", "x": ["www.jungewelt.de", "www.das-parlament.de", "www.juedische-allgemeine.de", "www.neues-deutschland.de", "www.bayernkurier.de", "www.boersen-zeitung.de", "www.die-tagespost.de", "www.generalanzeiger.de", "www.sonntagsblatt.de", "www.medical-tribune.de", "www.kath.net", "linkezeitung.de", "www.mittelbayerische.de", "www.architekturzeitung.com", "www.idowa.de", "jungefreiheit.de", "www.rhein-zeitung.de", "deutsche-wirtschafts-nachrichten.de", "www.epochtimes.de", "bnn.de", "www.berliner-zeitung.de", "taz.de", "www.saechsische.de", "www.sueddeutsche.de", "www.schwaebische.de", "www.aerztezeitung.de", "www.nw.de", "www.berliner-kurier.de", "www.morgenweb.de", "www.zeit.de", "www.infranken.de", "www.badische-zeitung.de", "www.rheinpfalz.de", "www.faz.net", "www.maz-online.de", "www.heise.de", "www.muensterschezeitung.de", "www.nordkurier.de", "www.kn-online.de", "www.rp-online.de", "www.svz.de", "www.tagesspiegel.de", "www.freitag.de", "www.wa.de", "www.mainpost.de", "www.bild.de", "www.onetz.de", "www.goettinger-tageblatt.de", "www.stimme.de", "www.neuepresse.de", "www.schwarzwaelder-bote.de", "www.ostsee-zeitung.de", "www.ovb-online.de", "www.volksstimme.de", "www.noz.de", "www.lr-online.de", "www.stuttgarter-nachrichten.de", "www.donaukurier.de", "www.stuttgarter-zeitung.de", "www.focus.de", "www.ln-online.de", "www.westfalen-blatt.de", "www.moz.de", "www.mz-web.de", "www.rnz.de", "www.lvz.de", "www.freiepresse.de", "www.wz.de", "www.main-echo.de", "www.azonline.de", "www.bz-berlin.de", "spiegel.de", "www.fr.de", "www.wiesbadener-kurier.de", "www.suedkurier.de", "www.nwzonline.de", "www.handelsblatt.com", "www.augsburger-allgemeine.de", "www.tz.de", "www.echo-online.de", "www.nordbayern.de", "www.welt.de", "www.kreiszeitung.de", "www.all-in.de", "www.swp.de", "www.merkur.de", "www.abendzeitung-muenchen.de", "www.aachener-nachrichten.de", "www.volksfreund.de", "www.general-anzeiger-bonn.de", "www.waz.de", "www.saarbruecker-zeitung.de"], "xaxis": "x", "y": [3, 1, 1, 6, 0, 0, 2, 32, 1, 0, 0, 0, 11, 0, 3, 5, 5, 5, 2, 2, 4, 2, 12, 43, 40, 15, 27, 4, 26, 16, 25, 12, 17, 12, 42, 4, 19, 35, 30, 36, 16, 12, 4, 27, 6, 18, 13, 31, 17, 29, 60, 35, 26, 6, 10, 7, 17, 7, 60, 10, 32, 18, 29, 39, 5, 43, 18, 33, 7, 22, 18, 12, 28, 38, 5, 9, 11, 3, 27, 41, 14, 18, 38, 6, 29, 26, 6, 49, 34, 32, 10, 33], "yaxis": "y"}, {"hovertemplate": "variable=third parties<br>index=%{x}<br>value=%{y}<extra></extra>", "legendgroup": "third parties", "line": {"color": "#EF553B", "dash": "solid"}, "mode": "lines", "name": "third parties", "orientation": "v", "showlegend": true, "type": "scatter", "x": ["www.jungewelt.de", "www.das-parlament.de", "www.juedische-allgemeine.de", "www.neues-deutschland.de", "www.bayernkurier.de", "www.boersen-zeitung.de", "www.die-tagespost.de", "www.generalanzeiger.de", "www.sonntagsblatt.de", "www.medical-tribune.de", "www.kath.net", "linkezeitung.de", "www.mittelbayerische.de", "www.architekturzeitung.com", "www.idowa.de", "jungefreiheit.de", "www.rhein-zeitung.de", "deutsche-wirtschafts-nachrichten.de", "www.epochtimes.de", "bnn.de", "www.berliner-zeitung.de", "taz.de", "www.saechsische.de", "www.sueddeutsche.de", "www.schwaebische.de", "www.aerztezeitung.de", "www.nw.de", "www.berliner-kurier.de", "www.morgenweb.de", "www.zeit.de", "www.infranken.de", "www.badische-zeitung.de", "www.rheinpfalz.de", "www.faz.net", "www.maz-online.de", "www.heise.de", "www.muensterschezeitung.de", "www.nordkurier.de", "www.kn-online.de", "www.rp-online.de", "www.svz.de", "www.tagesspiegel.de", "www.freitag.de", "www.wa.de", "www.mainpost.de", "www.bild.de", "www.onetz.de", "www.goettinger-tageblatt.de", "www.stimme.de", "www.neuepresse.de", "www.schwarzwaelder-bote.de", "www.ostsee-zeitung.de", "www.ovb-online.de", "www.volksstimme.de", "www.noz.de", "www.lr-online.de", "www.stuttgarter-nachrichten.de", "www.donaukurier.de", "www.stuttgarter-zeitung.de", "www.focus.de", "www.ln-online.de", "www.westfalen-blatt.de", "www.moz.de", "www.mz-web.de", "www.rnz.de", "www.lvz.de", "www.freiepresse.de", "www.wz.de", "www.main-echo.de", "www.azonline.de", "www.bz-berlin.de", "spiegel.de", "www.fr.de", "www.wiesbadener-kurier.de", "www.suedkurier.de", "www.nwzonline.de", "www.handelsblatt.com", "www.augsburger-allgemeine.de", "www.tz.de", "www.echo-online.de", "www.nordbayern.de", "www.welt.de", "www.kreiszeitung.de", "www.all-in.de", "www.swp.de", "www.merkur.de", "www.abendzeitung-muenchen.de", "www.aachener-nachrichten.de", "www.volksfreund.de", "www.general-anzeiger-bonn.de", "www.waz.de", "www.saarbruecker-zeitung.de"], "xaxis": "x", "y": [0, 1, 2, 4, 5, 10, 12, 12, 13, 14, 15, 16, 18, 20, 20, 20, 21, 22, 23, 23, 26, 27, 28, 34, 35, 37, 39, 41, 44, 45, 46, 49, 50, 52, 54, 54, 55, 56, 56, 56, 56, 59, 59, 60, 60, 61, 61, 62, 63, 63, 64, 64, 64, 64, 66, 66, 66, 67, 67, 67, 67, 69, 69, 69, 69, 70, 70, 70, 71, 71, 72, 72, 72, 74, 75, 75, 75, 77, 79, 79, 80, 82, 82, 82, 83, 85, 89, 90, 96, 105, 106, 107], "yaxis": "y"}],                        {"legend": {"title": {"text": "variable"}, "tracegroupgap": 0}, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "index"}}, "yaxis": {"anchor": "x", "domain": [0.0, 1.0], "title": {"text": "value"}}},                        {"responsive": true}                    ).then(function(){


                        })                };                });            </script>        </div>


It would be interesting to inspect individual ownerships related to individual third party companies but right now i'm still trying to figure out how to make a blog post with so much interactive data.

### third parties

The following table lists **all** third party servers from the dataset, which are over 500. I do not know about an ownership database for these kind of servers except search engines and [whois](https://en.wikipedia.org/wiki/WHOIS) servers. The latter ones do actually **not** allow anything with that data except checking for validity of domain names. In the course of privacy enhancements in the internet the whois-information is now largly hidden from public access. Especially in europe. Especially in germany. I align with the call for privacy of individuals in the face of evil *programmatic* exploitation by experts and fraudsters and all that, but why should they hide the registrant organization of `google.de`? That does not make me feel safer. Anyways, since the whois-data is now *redacted for privacy* i will just include what i got from the individual servers. 





<table border="1" class="dataframe compact" id="third-parties-table">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>registrant</th>
      <th>network</th>
      <th>websites</th>
      <th>websites %</th>
      <th>requests</th>
      <th>requests %</th>
      <th>article_referer</th>
      <th>bytes_sent</th>
      <th>bytes_received</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>doubleclick.net</th>
      <td>(US) Google Inc.</td>
      <td>(US) Google LLC</td>
      <td>86</td>
      <td>93.4783</td>
      <td>3453</td>
      <td>4.88036</td>
      <td>187</td>
      <td>40172</td>
      <td>107540109</td>
    </tr>
    <tr>
      <th>google.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>85</td>
      <td>92.3913</td>
      <td>1349</td>
      <td>1.90663</td>
      <td>119</td>
      <td>5274</td>
      <td>4706433</td>
    </tr>
    <tr>
      <th>google.de</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>84</td>
      <td>91.3043</td>
      <td>636</td>
      <td>0.898902</td>
      <td>204</td>
      <td>3387</td>
      <td>61170</td>
    </tr>
    <tr>
      <th>googlesyndication.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>82</td>
      <td>89.1304</td>
      <td>5820</td>
      <td>8.2258</td>
      <td>92</td>
      <td>35506</td>
      <td>93549606</td>
    </tr>
    <tr>
      <th>google-analytics.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>82</td>
      <td>89.1304</td>
      <td>1693</td>
      <td>2.39283</td>
      <td>24</td>
      <td>42808</td>
      <td>19755536</td>
    </tr>
    <tr>
      <th>googletagservices.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>81</td>
      <td>88.0435</td>
      <td>988</td>
      <td>1.39641</td>
      <td>6</td>
      <td>1058</td>
      <td>74692769</td>
    </tr>
    <tr>
      <th>googleapis.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>80</td>
      <td>86.9565</td>
      <td>480</td>
      <td>0.678416</td>
      <td>59</td>
      <td>2583</td>
      <td>26467641</td>
    </tr>
    <tr>
      <th>gstatic.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>78</td>
      <td>84.7826</td>
      <td>1360</td>
      <td>1.92218</td>
      <td>46</td>
      <td>8230</td>
      <td>27916889</td>
    </tr>
    <tr>
      <th>googletagmanager.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>73</td>
      <td>79.3478</td>
      <td>279</td>
      <td>0.39433</td>
      <td>93</td>
      <td>1420</td>
      <td>38415640</td>
    </tr>
    <tr>
      <th>ioam.de</th>
      <td></td>
      <td>(DE) INFOnline GmbH</td>
      <td>72</td>
      <td>78.2609</td>
      <td>554</td>
      <td>0.783006</td>
      <td>89</td>
      <td>22115</td>
      <td>6801898</td>
    </tr>
    <tr>
      <th>adnxs.com</th>
      <td>(US) AppNexus Inc</td>
      <td>(NL) AppNexus, Inc.</td>
      <td>69</td>
      <td>75</td>
      <td>1691</td>
      <td>2.39</td>
      <td>64</td>
      <td>69121</td>
      <td>26236152</td>
    </tr>
    <tr>
      <th>cloudfront.net</th>
      <td>(US) Amazon.com, Inc.</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>62</td>
      <td>67.3913</td>
      <td>601</td>
      <td>0.849434</td>
      <td>134</td>
      <td>13519</td>
      <td>8322958</td>
    </tr>
    <tr>
      <th>smartadserver.com</th>
      <td>(fr) Smartadserver</td>
      <td>(US) Sucuri</td>
      <td>60</td>
      <td>65.2174</td>
      <td>593</td>
      <td>0.838127</td>
      <td>43</td>
      <td>41971</td>
      <td>499934</td>
    </tr>
    <tr>
      <th>pubmatic.com</th>
      <td>(US) PubMatic, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>58</td>
      <td>63.0435</td>
      <td>766</td>
      <td>1.08264</td>
      <td>43</td>
      <td>2565</td>
      <td>10785118</td>
    </tr>
    <tr>
      <th>adsrvr.org</th>
      <td>(US) The Trade Desk</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>57</td>
      <td>61.9565</td>
      <td>312</td>
      <td>0.440971</td>
      <td>11</td>
      <td>38437</td>
      <td>41313</td>
    </tr>
    <tr>
      <th>amazon-adsystem.com</th>
      <td>(US) Amazon Technologies, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>56</td>
      <td>60.8696</td>
      <td>635</td>
      <td>0.897488</td>
      <td>25</td>
      <td>64867</td>
      <td>15117800</td>
    </tr>
    <tr>
      <th>casalemedia.com</th>
      <td>(CA) Index Exchange Inc.</td>
      <td>(CA) Comspec Communications</td>
      <td>56</td>
      <td>60.8696</td>
      <td>586</td>
      <td>0.828233</td>
      <td>8</td>
      <td>91229</td>
      <td>402461</td>
    </tr>
    <tr>
      <th>yieldlab.net</th>
      <td>(DE) Yieldlab AG</td>
      <td>(DE) myLoc managed IT AG</td>
      <td>54</td>
      <td>58.6957</td>
      <td>242</td>
      <td>0.342035</td>
      <td>16</td>
      <td>69775</td>
      <td>36620</td>
    </tr>
    <tr>
      <th>rubiconproject.com</th>
      <td>(US) The Rubicon Project, Inc.</td>
      <td>(US) Google LLC</td>
      <td>54</td>
      <td>58.6957</td>
      <td>721</td>
      <td>1.01904</td>
      <td>20</td>
      <td>21344</td>
      <td>3483276</td>
    </tr>
    <tr>
      <th>openx.net</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>54</td>
      <td>58.6957</td>
      <td>869</td>
      <td>1.22822</td>
      <td>21</td>
      <td>53415</td>
      <td>570619</td>
    </tr>
    <tr>
      <th>ampproject.org</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>53</td>
      <td>57.6087</td>
      <td>908</td>
      <td>1.28334</td>
      <td>72</td>
      <td>0</td>
      <td>52052365</td>
    </tr>
    <tr>
      <th>2mdn.net</th>
      <td>(US) Google Inc.</td>
      <td>(US) Google LLC</td>
      <td>48</td>
      <td>52.1739</td>
      <td>718</td>
      <td>1.0148</td>
      <td>0</td>
      <td>40</td>
      <td>32892092</td>
    </tr>
    <tr>
      <th>scorecardresearch.com</th>
      <td>(US) TMRG, Inc</td>
      <td>(US) CenturyLink Communications, LLC</td>
      <td>45</td>
      <td>48.913</td>
      <td>217</td>
      <td>0.306701</td>
      <td>14</td>
      <td>16747</td>
      <td>79011</td>
    </tr>
    <tr>
      <th>criteo.com</th>
      <td>(FR) Criteo SA</td>
      <td>(FR) Criteo Europe Infrastructures</td>
      <td>45</td>
      <td>48.913</td>
      <td>765</td>
      <td>1.08123</td>
      <td>108</td>
      <td>11019</td>
      <td>1992635</td>
    </tr>
    <tr>
      <th>adform.net</th>
      <td>(US) Savvy Investments, LLC Privacy ID# 10439376</td>
      <td>(DK) Adform DTC IPv4 Network</td>
      <td>44</td>
      <td>47.8261</td>
      <td>547</td>
      <td>0.773112</td>
      <td>17</td>
      <td>20413</td>
      <td>8666567</td>
    </tr>
    <tr>
      <th>xplosion.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>44</td>
      <td>47.8261</td>
      <td>264</td>
      <td>0.373129</td>
      <td>24</td>
      <td>63005</td>
      <td>515038</td>
    </tr>
    <tr>
      <th>mookie1.com</th>
      <td>(US) Xaxis</td>
      <td>(US) Google LLC</td>
      <td>44</td>
      <td>47.8261</td>
      <td>124</td>
      <td>0.175258</td>
      <td>1</td>
      <td>22925</td>
      <td>350797</td>
    </tr>
    <tr>
      <th>adsafeprotected.com</th>
      <td>(US) Integral Ad Science, Inc.</td>
      <td>(CA) Aptum Technologies</td>
      <td>43</td>
      <td>46.7391</td>
      <td>619</td>
      <td>0.874875</td>
      <td>19</td>
      <td>32507</td>
      <td>18214704</td>
    </tr>
    <tr>
      <th>mxcdn.net</th>
      <td>(DE)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>41</td>
      <td>44.5652</td>
      <td>158</td>
      <td>0.223312</td>
      <td>60</td>
      <td>0</td>
      <td>13322411</td>
    </tr>
    <tr>
      <th>outbrain.com</th>
      <td></td>
      <td>(US) Outbrain, Inc.</td>
      <td>41</td>
      <td>44.5652</td>
      <td>1468</td>
      <td>2.07482</td>
      <td>375</td>
      <td>37293</td>
      <td>33136032</td>
    </tr>
    <tr>
      <th>facebook.net</th>
      <td>(US) Facebook, Inc.</td>
      <td>(US) Facebook, Inc.</td>
      <td>40</td>
      <td>43.4783</td>
      <td>214</td>
      <td>0.302461</td>
      <td>6</td>
      <td>722</td>
      <td>31032898</td>
    </tr>
    <tr>
      <th>facebook.com</th>
      <td>(US) Facebook, Inc.</td>
      <td>(US) Facebook, Inc.</td>
      <td>40</td>
      <td>43.4783</td>
      <td>315</td>
      <td>0.445211</td>
      <td>7</td>
      <td>2122</td>
      <td>6319522</td>
    </tr>
    <tr>
      <th>outbrainimg.com</th>
      <td></td>
      <td>(US) Fastly</td>
      <td>39</td>
      <td>42.3913</td>
      <td>1392</td>
      <td>1.96741</td>
      <td>1145</td>
      <td>28942</td>
      <td>36550701</td>
    </tr>
    <tr>
      <th>adscale.de</th>
      <td></td>
      <td>(DE) Patrick Kambach</td>
      <td>38</td>
      <td>41.3043</td>
      <td>539</td>
      <td>0.761805</td>
      <td>35</td>
      <td>41070</td>
      <td>1798945</td>
    </tr>
    <tr>
      <th>meetrics.net</th>
      <td>(DE)</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>36</td>
      <td>39.1304</td>
      <td>839</td>
      <td>1.18582</td>
      <td>285</td>
      <td>44729</td>
      <td>36077</td>
    </tr>
    <tr>
      <th>media01.eu</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>36</td>
      <td>39.1304</td>
      <td>76</td>
      <td>0.107416</td>
      <td>0</td>
      <td>5718</td>
      <td>0</td>
    </tr>
    <tr>
      <th>indexww.com</th>
      <td>(CA) Index Exchange Inc.</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>36</td>
      <td>39.1304</td>
      <td>166</td>
      <td>0.234619</td>
      <td>77</td>
      <td>7</td>
      <td>295682</td>
    </tr>
    <tr>
      <th>medialead.de</th>
      <td></td>
      <td>(FR) EASY Marketing GmbH</td>
      <td>36</td>
      <td>39.1304</td>
      <td>127</td>
      <td>0.179498</td>
      <td>0</td>
      <td>2701</td>
      <td>18320</td>
    </tr>
    <tr>
      <th>privacy-mgmt.com</th>
      <td>(PA)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>35</td>
      <td>38.0435</td>
      <td>296</td>
      <td>0.418357</td>
      <td>48</td>
      <td>14961</td>
      <td>19372039</td>
    </tr>
    <tr>
      <th>nativendo.de</th>
      <td></td>
      <td>(DE) diva-e Datacenters GmbH</td>
      <td>35</td>
      <td>38.0435</td>
      <td>920</td>
      <td>1.3003</td>
      <td>281</td>
      <td>10130</td>
      <td>8174361</td>
    </tr>
    <tr>
      <th>jsdelivr.net</th>
      <td>(PA)</td>
      <td>(US) Fastly</td>
      <td>35</td>
      <td>38.0435</td>
      <td>73</td>
      <td>0.103176</td>
      <td>22</td>
      <td>240</td>
      <td>844910</td>
    </tr>
    <tr>
      <th>adition.com</th>
      <td>(DE) Virtual Minds AG</td>
      <td>(DE) HostPress GmbH, Kossmannstr. 7, 66571 Eppelborn</td>
      <td>34</td>
      <td>36.9565</td>
      <td>259</td>
      <td>0.366062</td>
      <td>4</td>
      <td>15059</td>
      <td>3926763</td>
    </tr>
    <tr>
      <th>otto.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>34</td>
      <td>36.9565</td>
      <td>272</td>
      <td>0.384436</td>
      <td>111</td>
      <td>0</td>
      <td>107863</td>
    </tr>
    <tr>
      <th>awin1.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>34</td>
      <td>36.9565</td>
      <td>132</td>
      <td>0.186565</td>
      <td>0</td>
      <td>2762</td>
      <td>265979</td>
    </tr>
    <tr>
      <th>criteo.net</th>
      <td>(FR) Criteo SA</td>
      <td>(FR) Criteo Europe Infrastructures</td>
      <td>33</td>
      <td>35.8696</td>
      <td>170</td>
      <td>0.240272</td>
      <td>9</td>
      <td>1176</td>
      <td>7372263</td>
    </tr>
    <tr>
      <th>dnacdn.net</th>
      <td>(FR) CRITEO SA</td>
      <td>(FR) Criteo Europe Infrastructures</td>
      <td>32</td>
      <td>34.7826</td>
      <td>130</td>
      <td>0.183738</td>
      <td>0</td>
      <td>3871</td>
      <td>4903</td>
    </tr>
    <tr>
      <th>taboola.com</th>
      <td>(DE)</td>
      <td>(US) Fastly</td>
      <td>31</td>
      <td>33.6957</td>
      <td>1772</td>
      <td>2.50449</td>
      <td>238</td>
      <td>33983</td>
      <td>88840637</td>
    </tr>
    <tr>
      <th>bidswitch.net</th>
      <td>(US)</td>
      <td>(GB) UKFast Admin</td>
      <td>31</td>
      <td>33.6957</td>
      <td>167</td>
      <td>0.236032</td>
      <td>10</td>
      <td>17594</td>
      <td>6493</td>
    </tr>
    <tr>
      <th>spotxchange.com</th>
      <td>(US) SpotX, Inc</td>
      <td>(US) SpotX, Inc.</td>
      <td>30</td>
      <td>32.6087</td>
      <td>262</td>
      <td>0.370302</td>
      <td>3</td>
      <td>15494</td>
      <td>5095</td>
    </tr>
    <tr>
      <th>mathtag.com</th>
      <td></td>
      <td>(US) MediaMath Inc</td>
      <td>30</td>
      <td>32.6087</td>
      <td>57</td>
      <td>0.080562</td>
      <td>0</td>
      <td>15481</td>
      <td>6952</td>
    </tr>
    <tr>
      <th>googleadservices.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>30</td>
      <td>32.6087</td>
      <td>78</td>
      <td>0.110243</td>
      <td>2</td>
      <td>4432</td>
      <td>157805</td>
    </tr>
    <tr>
      <th>redintelligence.net</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>28</td>
      <td>30.4348</td>
      <td>436</td>
      <td>0.616228</td>
      <td>1</td>
      <td>2851</td>
      <td>6353547</td>
    </tr>
    <tr>
      <th>id5-sync.com</th>
      <td></td>
      <td>(DE) OVH GmbH</td>
      <td>28</td>
      <td>30.4348</td>
      <td>38</td>
      <td>0.053708</td>
      <td>1</td>
      <td>24858</td>
      <td>6228</td>
    </tr>
    <tr>
      <th>360yield.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>27</td>
      <td>29.3478</td>
      <td>126</td>
      <td>0.178084</td>
      <td>0</td>
      <td>10036</td>
      <td>5629</td>
    </tr>
    <tr>
      <th>teads.tv</th>
      <td>(LU) Teads SA</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>25</td>
      <td>27.1739</td>
      <td>203</td>
      <td>0.286914</td>
      <td>0</td>
      <td>2765</td>
      <td>10234391</td>
    </tr>
    <tr>
      <th>ad-server.eu</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>25</td>
      <td>27.1739</td>
      <td>49</td>
      <td>0.069255</td>
      <td>0</td>
      <td>0</td>
      <td>23184</td>
    </tr>
    <tr>
      <th>advertising.com</th>
      <td>(US) Verizon Media Inc.</td>
      <td>(US) Oath Holdings Inc.</td>
      <td>25</td>
      <td>27.1739</td>
      <td>219</td>
      <td>0.309528</td>
      <td>7</td>
      <td>12216</td>
      <td>0</td>
    </tr>
    <tr>
      <th>theadex.com</th>
      <td>(DE) The ADEX GmbH</td>
      <td>(DE) HostPress GmbH, Kossmannstr. 7, 66571 Eppelborn</td>
      <td>24</td>
      <td>26.087</td>
      <td>421</td>
      <td>0.595028</td>
      <td>60</td>
      <td>13840</td>
      <td>1923018</td>
    </tr>
    <tr>
      <th>1rx.io</th>
      <td>(US) RhythmOne</td>
      <td>(NL) CUSTOMER-LAN</td>
      <td>24</td>
      <td>26.087</td>
      <td>209</td>
      <td>0.295394</td>
      <td>1</td>
      <td>3599</td>
      <td>258</td>
    </tr>
    <tr>
      <th>cleverpush.com</th>
      <td>(PA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>24</td>
      <td>26.087</td>
      <td>177</td>
      <td>0.250166</td>
      <td>84</td>
      <td>458</td>
      <td>29352102</td>
    </tr>
    <tr>
      <th>fastly.net</th>
      <td>(US) DNStination Inc</td>
      <td>(US) Fastly</td>
      <td>23</td>
      <td>25</td>
      <td>137</td>
      <td>0.193631</td>
      <td>59</td>
      <td>323</td>
      <td>5356970</td>
    </tr>
    <tr>
      <th>f11-ads.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>23</td>
      <td>25</td>
      <td>410</td>
      <td>0.579481</td>
      <td>119</td>
      <td>5812</td>
      <td>5885858</td>
    </tr>
    <tr>
      <th>yahoo.com</th>
      <td>(US) Oath Inc.</td>
      <td>(US) Oath Holdings Inc.</td>
      <td>22</td>
      <td>23.913</td>
      <td>38</td>
      <td>0.053708</td>
      <td>1</td>
      <td>14702</td>
      <td>10914</td>
    </tr>
    <tr>
      <th>vgwort.de</th>
      <td></td>
      <td>(DE) Neue Medien Muennich GmbH</td>
      <td>21</td>
      <td>22.8261</td>
      <td>42</td>
      <td>0.0593614</td>
      <td>38</td>
      <td>0</td>
      <td>1806</td>
    </tr>
    <tr>
      <th>semasio.net</th>
      <td>(DE)</td>
      <td>(DK) Netic A/S</td>
      <td>21</td>
      <td>22.8261</td>
      <td>80</td>
      <td>0.113069</td>
      <td>18</td>
      <td>21009</td>
      <td>3360</td>
    </tr>
    <tr>
      <th>cdntrf.com</th>
      <td>(PA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>21</td>
      <td>22.8261</td>
      <td>181</td>
      <td>0.25582</td>
      <td>84</td>
      <td>0</td>
      <td>26885721</td>
    </tr>
    <tr>
      <th>emetriq.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>20</td>
      <td>21.7391</td>
      <td>36</td>
      <td>0.0508812</td>
      <td>14</td>
      <td>2278</td>
      <td>31248</td>
    </tr>
    <tr>
      <th>webgains.com</th>
      <td>(GB) Webgains Ltd</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>20</td>
      <td>21.7391</td>
      <td>159</td>
      <td>0.224725</td>
      <td>0</td>
      <td>7002</td>
      <td>553514</td>
    </tr>
    <tr>
      <th>chartbeat.net</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>20</td>
      <td>21.7391</td>
      <td>140</td>
      <td>0.197871</td>
      <td>0</td>
      <td>10063</td>
      <td>4988</td>
    </tr>
    <tr>
      <th>chartbeat.com</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>20</td>
      <td>21.7391</td>
      <td>109</td>
      <td>0.154057</td>
      <td>0</td>
      <td>1484</td>
      <td>2418102</td>
    </tr>
    <tr>
      <th>prebid.org</th>
      <td>(US)</td>
      <td>(US) Pantheon</td>
      <td>20</td>
      <td>21.7391</td>
      <td>38</td>
      <td>0.053708</td>
      <td>18</td>
      <td>0</td>
      <td>51022</td>
    </tr>
    <tr>
      <th>webgains.io</th>
      <td>(GB) Webgains</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>20</td>
      <td>21.7391</td>
      <td>149</td>
      <td>0.210592</td>
      <td>0</td>
      <td>0</td>
      <td>4045062</td>
    </tr>
    <tr>
      <th>lp4.io</th>
      <td>(US)</td>
      <td>(NO) GLOBALCONNECT AS</td>
      <td>20</td>
      <td>21.7391</td>
      <td>204</td>
      <td>0.288327</td>
      <td>73</td>
      <td>5037</td>
      <td>3292737</td>
    </tr>
    <tr>
      <th>justpremium.com</th>
      <td>(NL)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>346</td>
      <td>0.489025</td>
      <td>89</td>
      <td>4073</td>
      <td>9537495</td>
    </tr>
    <tr>
      <th>trmcdn.eu</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>142</td>
      <td>0.200698</td>
      <td>11</td>
      <td>22</td>
      <td>12599930</td>
    </tr>
    <tr>
      <th>yieldlove-ad-serving.net</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>352</td>
      <td>0.497505</td>
      <td>116</td>
      <td>0</td>
      <td>1722</td>
    </tr>
    <tr>
      <th>lead-alliance.net</th>
      <td>(DE) lead alliance GmbH</td>
      <td>(DE) Bloonix GmbH</td>
      <td>19</td>
      <td>20.6522</td>
      <td>59</td>
      <td>0.0833887</td>
      <td>0</td>
      <td>2407</td>
      <td>17169</td>
    </tr>
    <tr>
      <th>twitter.com</th>
      <td>(US) Twitter, Inc.</td>
      <td>(US) Twitter Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>167</td>
      <td>0.236032</td>
      <td>4</td>
      <td>2835</td>
      <td>22764247</td>
    </tr>
    <tr>
      <th>userreport.com</th>
      <td>(DK) AudienceProject A/S</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>161</td>
      <td>0.227552</td>
      <td>51</td>
      <td>13617</td>
      <td>4778490</td>
    </tr>
    <tr>
      <th>m-t.io</th>
      <td>(GB) Tech Essence Limited</td>
      <td>(US) Squarespace, Inc.</td>
      <td>19</td>
      <td>20.6522</td>
      <td>76</td>
      <td>0.107416</td>
      <td>0</td>
      <td>2976</td>
      <td>2645</td>
    </tr>
    <tr>
      <th>telefonica-partner.de</th>
      <td></td>
      <td>(DE) Bloonix GmbH</td>
      <td>18</td>
      <td>19.5652</td>
      <td>57</td>
      <td>0.080562</td>
      <td>0</td>
      <td>2382</td>
      <td>2365</td>
    </tr>
    <tr>
      <th>sascdn.com</th>
      <td>(fr) Smartadserver</td>
      <td>(FR) SafeBrands S.A.S.</td>
      <td>18</td>
      <td>19.5652</td>
      <td>153</td>
      <td>0.216245</td>
      <td>1</td>
      <td>0</td>
      <td>3616931</td>
    </tr>
    <tr>
      <th>blau.de</th>
      <td></td>
      <td>(DE) o2 Germany GmbH &amp; Co. OHG</td>
      <td>18</td>
      <td>19.5652</td>
      <td>61</td>
      <td>0.0862154</td>
      <td>0</td>
      <td>4621</td>
      <td>2537</td>
    </tr>
    <tr>
      <th>transmatico.com</th>
      <td>(DE)</td>
      <td>(DE) digitalocean</td>
      <td>18</td>
      <td>19.5652</td>
      <td>57</td>
      <td>0.080562</td>
      <td>4</td>
      <td>2649</td>
      <td>3454429</td>
    </tr>
    <tr>
      <th>m6r.eu</th>
      <td></td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>18</td>
      <td>19.5652</td>
      <td>91</td>
      <td>0.128616</td>
      <td>21</td>
      <td>15217</td>
      <td>19417</td>
    </tr>
    <tr>
      <th>ad4m.at</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>18</td>
      <td>19.5652</td>
      <td>337</td>
      <td>0.476305</td>
      <td>0</td>
      <td>935</td>
      <td>7904111</td>
    </tr>
    <tr>
      <th>purelocalmedia.de</th>
      <td></td>
      <td>(DE) Strato AG</td>
      <td>18</td>
      <td>19.5652</td>
      <td>133</td>
      <td>0.187978</td>
      <td>68</td>
      <td>19587</td>
      <td>873774</td>
    </tr>
    <tr>
      <th>ad.gt</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>18</td>
      <td>19.5652</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>0</td>
      <td>1060</td>
      <td>774</td>
    </tr>
    <tr>
      <th>ad4mat.net</th>
      <td>(CA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>17</td>
      <td>18.4783</td>
      <td>37</td>
      <td>0.0522946</td>
      <td>8</td>
      <td>0</td>
      <td>51652</td>
    </tr>
    <tr>
      <th>lijit.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>17</td>
      <td>18.4783</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>2</td>
      <td>13908</td>
      <td>385</td>
    </tr>
    <tr>
      <th>stroeerdigitalgroup.de</th>
      <td></td>
      <td>(DE) InterNetX GmbH</td>
      <td>17</td>
      <td>18.4783</td>
      <td>64</td>
      <td>0.0904555</td>
      <td>26</td>
      <td>0</td>
      <td>21128028</td>
    </tr>
    <tr>
      <th>ad-production-stage.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>17</td>
      <td>18.4783</td>
      <td>708</td>
      <td>1.00066</td>
      <td>11</td>
      <td>0</td>
      <td>36391734</td>
    </tr>
    <tr>
      <th>everesttech.net</th>
      <td>(US) Adobe Inc.</td>
      <td>(US) Adobe Inc.</td>
      <td>16</td>
      <td>17.3913</td>
      <td>37</td>
      <td>0.0522946</td>
      <td>7</td>
      <td>12481</td>
      <td>2844</td>
    </tr>
    <tr>
      <th>o2online.de</th>
      <td></td>
      <td>(DE) o2 Germany GmbH &amp; Co. OHG</td>
      <td>16</td>
      <td>17.3913</td>
      <td>53</td>
      <td>0.0749085</td>
      <td>0</td>
      <td>4332</td>
      <td>2236</td>
    </tr>
    <tr>
      <th>rqtrk.eu</th>
      <td></td>
      <td>(DE) OVH GmbH</td>
      <td>16</td>
      <td>17.3913</td>
      <td>23</td>
      <td>0.0325075</td>
      <td>0</td>
      <td>16152</td>
      <td>828</td>
    </tr>
    <tr>
      <th>amazonaws.com</th>
      <td>(US) Amazon.com, Inc.</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>16</td>
      <td>17.3913</td>
      <td>79</td>
      <td>0.111656</td>
      <td>7</td>
      <td>338</td>
      <td>1927079</td>
    </tr>
    <tr>
      <th>bttrack.com</th>
      <td>(GB)</td>
      <td>(US) Bidtellect Inc.</td>
      <td>16</td>
      <td>17.3913</td>
      <td>16</td>
      <td>0.0226139</td>
      <td>0</td>
      <td>3590</td>
      <td>105</td>
    </tr>
    <tr>
      <th>exactag.com</th>
      <td>(DE)</td>
      <td>(DE) conversis GmbH</td>
      <td>16</td>
      <td>17.3913</td>
      <td>21</td>
      <td>0.0296807</td>
      <td>5</td>
      <td>6648</td>
      <td>23115</td>
    </tr>
    <tr>
      <th>recognified.net</th>
      <td>(DE) Online Solution Int Ltd</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>15</td>
      <td>16.3043</td>
      <td>60</td>
      <td>0.0848021</td>
      <td>48</td>
      <td>3852</td>
      <td>2823824</td>
    </tr>
    <tr>
      <th>stroeerdigital.de</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>15</td>
      <td>16.3043</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>10</td>
      <td>0</td>
      <td>57219</td>
    </tr>
    <tr>
      <th>videoreach.com</th>
      <td>(DE)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>15</td>
      <td>16.3043</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>25</td>
      <td>17380</td>
      <td>38906</td>
    </tr>
    <tr>
      <th>emxdgt.com</th>
      <td>(US) Engine</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>130</td>
      <td>0.183738</td>
      <td>13</td>
      <td>683</td>
      <td>86973</td>
    </tr>
    <tr>
      <th>mfadsrvr.com</th>
      <td>(IL)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>20</td>
      <td>0.0282674</td>
      <td>0</td>
      <td>1766</td>
      <td>258</td>
    </tr>
    <tr>
      <th>contextweb.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>34</td>
      <td>0.0480545</td>
      <td>0</td>
      <td>2516</td>
      <td>258</td>
    </tr>
    <tr>
      <th>cloudflare.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>46</td>
      <td>0.0650149</td>
      <td>13</td>
      <td>6</td>
      <td>1591093</td>
    </tr>
    <tr>
      <th>cheqzone.com</th>
      <td>(IL) cheq.ai</td>
      <td>(DE) CDN77 Frankfurt - Bunny CDN</td>
      <td>14</td>
      <td>15.2174</td>
      <td>86</td>
      <td>0.12155</td>
      <td>35</td>
      <td>13565</td>
      <td>1488678</td>
    </tr>
    <tr>
      <th>adkernel.com</th>
      <td>(US) Adkernel, LLC</td>
      <td>(US) Webair Internet Development Company Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>14</td>
      <td>0.0197871</td>
      <td>0</td>
      <td>1092</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tremorhub.com</th>
      <td>(US) Telaria</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>14</td>
      <td>15.2174</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>0</td>
      <td>24541</td>
      <td>1247</td>
    </tr>
    <tr>
      <th>serving-sys.com</th>
      <td>(US) Andreas Acquisition LLC</td>
      <td>(GB) TeleCity Group Customer - Sizmek</td>
      <td>13</td>
      <td>14.1304</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>0</td>
      <td>2096</td>
      <td>31428</td>
    </tr>
    <tr>
      <th>zemanta.com</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>13</td>
      <td>14.1304</td>
      <td>46</td>
      <td>0.0650149</td>
      <td>0</td>
      <td>1619</td>
      <td>36184</td>
    </tr>
    <tr>
      <th>exelator.com</th>
      <td>(US) The Nielsen Company</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>13</td>
      <td>14.1304</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>3</td>
      <td>949</td>
      <td>0</td>
    </tr>
    <tr>
      <th>appier.net</th>
      <td>(SG) Appier Pte. Ltd.</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>13</td>
      <td>14.1304</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>stickyadstv.com</th>
      <td>(FR) STICKY ADS TV S.A.S.</td>
      <td>(FR) OVH SAS</td>
      <td>13</td>
      <td>14.1304</td>
      <td>104</td>
      <td>0.14699</td>
      <td>16</td>
      <td>3912</td>
      <td>3125553</td>
    </tr>
    <tr>
      <th>vidazoo.com</th>
      <td>(IL) Vidazoo Ltd</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>13</td>
      <td>14.1304</td>
      <td>392</td>
      <td>0.55404</td>
      <td>340</td>
      <td>112</td>
      <td>25220402</td>
    </tr>
    <tr>
      <th>opencmp.net</th>
      <td>(US)</td>
      <td>(DE) dogado GmbH</td>
      <td>13</td>
      <td>14.1304</td>
      <td>156</td>
      <td>0.220485</td>
      <td>65</td>
      <td>0</td>
      <td>15430612</td>
    </tr>
    <tr>
      <th>plista.com</th>
      <td>(DE) plista GmbH</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>12</td>
      <td>13.0435</td>
      <td>170</td>
      <td>0.240272</td>
      <td>9</td>
      <td>5568</td>
      <td>4360795</td>
    </tr>
    <tr>
      <th>weekli.systems</th>
      <td>(DE) mcosys GmbH</td>
      <td>(DE) mcosys GmbH</td>
      <td>12</td>
      <td>13.0435</td>
      <td>143</td>
      <td>0.202112</td>
      <td>2</td>
      <td>62</td>
      <td>3078102</td>
    </tr>
    <tr>
      <th>demdex.net</th>
      <td>(US) Adobe Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>12</td>
      <td>13.0435</td>
      <td>54</td>
      <td>0.0763219</td>
      <td>2</td>
      <td>1696</td>
      <td>93495</td>
    </tr>
    <tr>
      <th>podigee.io</th>
      <td>(DE) Podigee GmbH</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>12</td>
      <td>13.0435</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>0</td>
      <td>363</td>
      <td>1496569</td>
    </tr>
    <tr>
      <th>opinary.com</th>
      <td>(PA)</td>
      <td>(DE) HEG Mass</td>
      <td>12</td>
      <td>13.0435</td>
      <td>82</td>
      <td>0.115896</td>
      <td>27</td>
      <td>751</td>
      <td>770722</td>
    </tr>
    <tr>
      <th>podigee-cdn.net</th>
      <td>(DE)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>12</td>
      <td>13.0435</td>
      <td>88</td>
      <td>0.124376</td>
      <td>9</td>
      <td>18</td>
      <td>8749750</td>
    </tr>
    <tr>
      <th>twiago.com</th>
      <td></td>
      <td>(GB) DFL-NET</td>
      <td>12</td>
      <td>13.0435</td>
      <td>213</td>
      <td>0.301047</td>
      <td>90</td>
      <td>1022</td>
      <td>1834433</td>
    </tr>
    <tr>
      <th>sp-prod.net</th>
      <td>(PA)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>12</td>
      <td>13.0435</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>10</td>
      <td>0</td>
      <td>4286072</td>
    </tr>
    <tr>
      <th>technoratimedia.com</th>
      <td>(US) Synacor, Inc.</td>
      <td>(US) Synacor, Inc.</td>
      <td>11</td>
      <td>11.9565</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>1969</td>
      <td>0</td>
    </tr>
    <tr>
      <th>onetag-sys.com</th>
      <td>(GB) CrossReactive LTD</td>
      <td>(DE) OVH GmbH</td>
      <td>11</td>
      <td>11.9565</td>
      <td>67</td>
      <td>0.0946956</td>
      <td>32</td>
      <td>90</td>
      <td>58453</td>
    </tr>
    <tr>
      <th>dwcdn.net</th>
      <td>(US)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>11</td>
      <td>11.9565</td>
      <td>234</td>
      <td>0.330728</td>
      <td>2</td>
      <td>42</td>
      <td>22377324</td>
    </tr>
    <tr>
      <th>sonobi.com</th>
      <td>(US) Sonobi, Inc</td>
      <td>(US) Google LLC</td>
      <td>11</td>
      <td>11.9565</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>1320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>aniview.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>11</td>
      <td>11.9565</td>
      <td>291</td>
      <td>0.41129</td>
      <td>273</td>
      <td>4919</td>
      <td>19411300</td>
    </tr>
    <tr>
      <th>podigee.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>11</td>
      <td>11.9565</td>
      <td>31</td>
      <td>0.0438144</td>
      <td>7</td>
      <td>125</td>
      <td>4755630</td>
    </tr>
    <tr>
      <th>adrtx.net</th>
      <td>(DE)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>11</td>
      <td>11.9565</td>
      <td>36</td>
      <td>0.0508812</td>
      <td>0</td>
      <td>165</td>
      <td>15822</td>
    </tr>
    <tr>
      <th>visx.net</th>
      <td>(DE) YOC AG</td>
      <td>(US) Google LLC</td>
      <td>11</td>
      <td>11.9565</td>
      <td>67</td>
      <td>0.0946956</td>
      <td>28</td>
      <td>6680</td>
      <td>1311</td>
    </tr>
    <tr>
      <th>consensu.org</th>
      <td>(BE) IAB Europe</td>
      <td>(US) Google LLC</td>
      <td>11</td>
      <td>11.9565</td>
      <td>175</td>
      <td>0.247339</td>
      <td>56</td>
      <td>1450</td>
      <td>7956314</td>
    </tr>
    <tr>
      <th>weekli.de</th>
      <td></td>
      <td>(DE) mcosys GmbH</td>
      <td>11</td>
      <td>11.9565</td>
      <td>37</td>
      <td>0.0522946</td>
      <td>0</td>
      <td>768</td>
      <td>426866</td>
    </tr>
    <tr>
      <th>datawrapper.de</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>11</td>
      <td>11.9565</td>
      <td>20</td>
      <td>0.0282674</td>
      <td>0</td>
      <td>46</td>
      <td>860</td>
    </tr>
    <tr>
      <th>doubleverify.com</th>
      <td>(US) DoubleVerify</td>
      <td>(US) Unified Layer</td>
      <td>10</td>
      <td>10.8696</td>
      <td>40</td>
      <td>0.0565347</td>
      <td>1</td>
      <td>1676</td>
      <td>1111373</td>
    </tr>
    <tr>
      <th>perfectmarket.com</th>
      <td>(US) Taboola, Inc</td>
      <td>(US) Media Temple, Inc.</td>
      <td>10</td>
      <td>10.8696</td>
      <td>34</td>
      <td>0.0480545</td>
      <td>8</td>
      <td>0</td>
      <td>2031828</td>
    </tr>
    <tr>
      <th>polyfill.io</th>
      <td>(GB) The Financial Times Limited</td>
      <td>(US) Fastly</td>
      <td>10</td>
      <td>10.8696</td>
      <td>48</td>
      <td>0.0678416</td>
      <td>10</td>
      <td>360</td>
      <td>7686</td>
    </tr>
    <tr>
      <th>office-partner.de</th>
      <td></td>
      <td>(DE) SysEleven GmbH</td>
      <td>10</td>
      <td>10.8696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>0</td>
      <td>380</td>
      <td>17413</td>
    </tr>
    <tr>
      <th>cxense.com</th>
      <td>(US) Piano Software</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>10</td>
      <td>10.8696</td>
      <td>211</td>
      <td>0.298221</td>
      <td>13</td>
      <td>3749</td>
      <td>4329922</td>
    </tr>
    <tr>
      <th>glomex.com</th>
      <td>(DE)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>10</td>
      <td>10.8696</td>
      <td>322</td>
      <td>0.455104</td>
      <td>42</td>
      <td>7995</td>
      <td>6481377</td>
    </tr>
    <tr>
      <th>dspx.tv</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>9</td>
      <td>9.78261</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>9</td>
      <td>7306</td>
      <td>48963</td>
    </tr>
    <tr>
      <th>glomex.cloud</th>
      <td>(DE) Glomex GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>9</td>
      <td>9.78261</td>
      <td>212</td>
      <td>0.299634</td>
      <td>46</td>
      <td>637</td>
      <td>87452426</td>
    </tr>
    <tr>
      <th>smartclip.net</th>
      <td>(DE)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>9</td>
      <td>9.78261</td>
      <td>54</td>
      <td>0.0763219</td>
      <td>24</td>
      <td>1118</td>
      <td>800829</td>
    </tr>
    <tr>
      <th>disqus.com</th>
      <td>(US) Disqus, Inc.</td>
      <td>(US) Fastly</td>
      <td>8</td>
      <td>8.69565</td>
      <td>42</td>
      <td>0.0593614</td>
      <td>33</td>
      <td>54</td>
      <td>255392</td>
    </tr>
    <tr>
      <th>boltdns.net</th>
      <td>(US)</td>
      <td>(US) Fastly</td>
      <td>8</td>
      <td>8.69565</td>
      <td>52</td>
      <td>0.0734951</td>
      <td>0</td>
      <td>142</td>
      <td>3584040</td>
    </tr>
    <tr>
      <th>bluekai.com</th>
      <td>(US)</td>
      <td>(US) Oracle Corporation</td>
      <td>8</td>
      <td>8.69565</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>2981</td>
      <td>26</td>
    </tr>
    <tr>
      <th>adsafety.net</th>
      <td>(DE)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>8</td>
      <td>8.69565</td>
      <td>49</td>
      <td>0.069255</td>
      <td>23</td>
      <td>295</td>
      <td>7152</td>
    </tr>
    <tr>
      <th>oadts.com</th>
      <td>(DE) ATG Ad Tech Group GmbH</td>
      <td>(DE) Wavecon GmbH</td>
      <td>8</td>
      <td>8.69565</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>21</td>
      <td>9363</td>
      <td>395954</td>
    </tr>
    <tr>
      <th>de.com</th>
      <td>(UK)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>8</td>
      <td>8.69565</td>
      <td>127</td>
      <td>0.179498</td>
      <td>11</td>
      <td>3523</td>
      <td>282891</td>
    </tr>
    <tr>
      <th>f11-ads.net</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>8</td>
      <td>8.69565</td>
      <td>52</td>
      <td>0.0734951</td>
      <td>32</td>
      <td>14783</td>
      <td>798579</td>
    </tr>
    <tr>
      <th>onetrust.com</th>
      <td>(PA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>8</td>
      <td>8.69565</td>
      <td>30</td>
      <td>0.042401</td>
      <td>5</td>
      <td>0</td>
      <td>38288</td>
    </tr>
    <tr>
      <th>ytimg.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>8</td>
      <td>8.69565</td>
      <td>26</td>
      <td>0.0367476</td>
      <td>0</td>
      <td>0</td>
      <td>633045</td>
    </tr>
    <tr>
      <th>ml314.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>875</td>
      <td>0</td>
    </tr>
    <tr>
      <th>adobedtm.com</th>
      <td>(US) Adobe Inc.</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>86</td>
      <td>0.12155</td>
      <td>17</td>
      <td>0</td>
      <td>5284195</td>
    </tr>
    <tr>
      <th>madsack-native.de</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>7</td>
      <td>7.6087</td>
      <td>54</td>
      <td>0.0763219</td>
      <td>0</td>
      <td>492</td>
      <td>624148</td>
    </tr>
    <tr>
      <th>haz.de</th>
      <td></td>
      <td>(DE) Verlagsgesellschaft Madsack GmbH &amp; Co.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>16</td>
      <td>0.0226139</td>
      <td>0</td>
      <td>0</td>
      <td>340384</td>
    </tr>
    <tr>
      <th>agkn.com</th>
      <td>(US) Neustar, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>455</td>
      <td>0</td>
    </tr>
    <tr>
      <th>rndtech.de</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>665</td>
      <td>0.939889</td>
      <td>27</td>
      <td>0</td>
      <td>13055046</td>
    </tr>
    <tr>
      <th>instagram.com</th>
      <td>(US) Instagram LLC</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>31</td>
      <td>0.0438144</td>
      <td>10</td>
      <td>0</td>
      <td>484127</td>
    </tr>
    <tr>
      <th>youtube.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>7</td>
      <td>7.6087</td>
      <td>147</td>
      <td>0.207765</td>
      <td>0</td>
      <td>1772</td>
      <td>29393519</td>
    </tr>
    <tr>
      <th>sphere.com</th>
      <td></td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>121</td>
      <td>0.171017</td>
      <td>0</td>
      <td>0</td>
      <td>5734795</td>
    </tr>
    <tr>
      <th>liadm.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>511</td>
      <td>0</td>
    </tr>
    <tr>
      <th>crwdcntrl.net</th>
      <td>(US)</td>
      <td>(US) Lotame Solutions, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>mlsat02.de</th>
      <td></td>
      <td>(FR) OVH SAS</td>
      <td>7</td>
      <td>7.6087</td>
      <td>24</td>
      <td>0.0339208</td>
      <td>0</td>
      <td>369</td>
      <td>16492</td>
    </tr>
    <tr>
      <th>unrulymedia.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>0</td>
      <td>2865</td>
      <td>33921</td>
    </tr>
    <tr>
      <th>opecloud.com</th>
      <td>(CH) 1plusX AG</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>44</td>
      <td>0.0621882</td>
      <td>13</td>
      <td>4183</td>
      <td>28882</td>
    </tr>
    <tr>
      <th>zeotap.com</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>0</td>
      <td>6040</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cookielaw.org</th>
      <td>(PA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>7</td>
      <td>7.6087</td>
      <td>123</td>
      <td>0.173844</td>
      <td>40</td>
      <td>0</td>
      <td>14940745</td>
    </tr>
    <tr>
      <th>smartstream.tv</th>
      <td></td>
      <td>(DE) OVH GmbH</td>
      <td>7</td>
      <td>7.6087</td>
      <td>47</td>
      <td>0.0664283</td>
      <td>1</td>
      <td>8303</td>
      <td>81843</td>
    </tr>
    <tr>
      <th>pushwoosh.com</th>
      <td>(NZ) Arello Mobile</td>
      <td>(ZZ) APNIC-STUB</td>
      <td>7</td>
      <td>7.6087</td>
      <td>113</td>
      <td>0.159711</td>
      <td>54</td>
      <td>0</td>
      <td>5703273</td>
    </tr>
    <tr>
      <th>akamaihd.net</th>
      <td>(US) Akamai Technologies, inc.</td>
      <td>(DE) Telefonica Germany GmbH &amp; Co. OHG</td>
      <td>7</td>
      <td>7.6087</td>
      <td>50</td>
      <td>0.0706684</td>
      <td>2</td>
      <td>438</td>
      <td>21894460</td>
    </tr>
    <tr>
      <th>showheroes.com</th>
      <td>(DE) ShowHeroes GmbH</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>6</td>
      <td>6.52174</td>
      <td>81</td>
      <td>0.114483</td>
      <td>51</td>
      <td>3366</td>
      <td>4823990</td>
    </tr>
    <tr>
      <th>ibillboard.com</th>
      <td>(CZ) Internet BillBoard a.s.</td>
      <td>(CZ) Internet BillBoard a.s.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>11</td>
      <td>0.015547</td>
      <td>6</td>
      <td>546</td>
      <td>0</td>
    </tr>
    <tr>
      <th>googlevideo.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>6</td>
      <td>6.52174</td>
      <td>16</td>
      <td>0.0226139</td>
      <td>0</td>
      <td>3586</td>
      <td>15736828</td>
    </tr>
    <tr>
      <th>turn.com</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>6</td>
      <td>6.52174</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>0</td>
      <td>4553</td>
      <td>46</td>
    </tr>
    <tr>
      <th>aticdn.net</th>
      <td>(FR) Applied Technologies Internet SAS</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>6</td>
      <td>0</td>
      <td>618942</td>
    </tr>
    <tr>
      <th>trmads.eu</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>67</td>
      <td>0.0946956</td>
      <td>18</td>
      <td>663</td>
      <td>12500359</td>
    </tr>
    <tr>
      <th>usercentrics.eu</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>6</td>
      <td>6.52174</td>
      <td>213</td>
      <td>0.301047</td>
      <td>67</td>
      <td>0</td>
      <td>13762583</td>
    </tr>
    <tr>
      <th>bing.com</th>
      <td>(US) Microsoft Corporation</td>
      <td>(US) Microsoft Corporation</td>
      <td>6</td>
      <td>6.52174</td>
      <td>37</td>
      <td>0.0522946</td>
      <td>12</td>
      <td>2507</td>
      <td>373529</td>
    </tr>
    <tr>
      <th>telekom.de</th>
      <td></td>
      <td>(DE) T-Systems International GmbH</td>
      <td>6</td>
      <td>6.52174</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>1482</td>
      <td>301</td>
    </tr>
    <tr>
      <th>omtrdc.net</th>
      <td>(US) Adobe Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>15</td>
      <td>0.0212005</td>
      <td>1</td>
      <td>1022</td>
      <td>31127</td>
    </tr>
    <tr>
      <th>ggpht.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>6</td>
      <td>6.52174</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>0</td>
      <td>26322</td>
    </tr>
    <tr>
      <th>3lift.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>4</td>
      <td>5868</td>
      <td>152</td>
    </tr>
    <tr>
      <th>idcdn.de</th>
      <td></td>
      <td>(DE) Ippen Digital GmbH &amp; Co. KG</td>
      <td>6</td>
      <td>6.52174</td>
      <td>285</td>
      <td>0.40281</td>
      <td>139</td>
      <td>0</td>
      <td>2491269</td>
    </tr>
    <tr>
      <th>ippen.space</th>
      <td>(DE) Ippen Digital GmbH &amp; Co. KG</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>33</td>
      <td>0.0466411</td>
      <td>11</td>
      <td>0</td>
      <td>331597</td>
    </tr>
    <tr>
      <th>xiti.com</th>
      <td>(FR) Applied Technologies Internet SAS</td>
      <td>(FR) AT INTERNET Network Team</td>
      <td>6</td>
      <td>6.52174</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>6</td>
      <td>3435</td>
      <td>980</td>
    </tr>
    <tr>
      <th>geoedge.be</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>6</td>
      <td>6.52174</td>
      <td>116</td>
      <td>0.163951</td>
      <td>40</td>
      <td>0</td>
      <td>5801128</td>
    </tr>
    <tr>
      <th>reisereporter.de</th>
      <td></td>
      <td>(DE) Verlagsgesellschaft Madsack GmbH &amp; Co.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>20</td>
      <td>0.0282674</td>
      <td>0</td>
      <td>20</td>
      <td>384314</td>
    </tr>
    <tr>
      <th>brillen.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>4</td>
      <td>265</td>
      <td>344</td>
    </tr>
    <tr>
      <th>yieldscale.com</th>
      <td>(DE)</td>
      <td>(DE) Neue Medien Muennich GmbH</td>
      <td>5</td>
      <td>5.43478</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>6</td>
      <td>0</td>
      <td>4118992</td>
    </tr>
    <tr>
      <th>nuggad.net</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>3880</td>
      <td>5331</td>
    </tr>
    <tr>
      <th>artefact.com</th>
      <td>(FR) ARTEFACT</td>
      <td>(FR) OVH SAS</td>
      <td>5</td>
      <td>5.43478</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>290</td>
      <td>0</td>
    </tr>
    <tr>
      <th>tinypass.com</th>
      <td>Piano Software</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>114</td>
      <td>0.161124</td>
      <td>26</td>
      <td>1713</td>
      <td>11362571</td>
    </tr>
    <tr>
      <th>wetterkontor.de</th>
      <td></td>
      <td>(DE) Strato AG</td>
      <td>5</td>
      <td>5.43478</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>0</td>
      <td>28057</td>
    </tr>
    <tr>
      <th>pressekompass.net</th>
      <td>(DE)</td>
      <td>(DE) SHARED WEBHOSTING</td>
      <td>5</td>
      <td>5.43478</td>
      <td>68</td>
      <td>0.096109</td>
      <td>1</td>
      <td>1732</td>
      <td>6025805</td>
    </tr>
    <tr>
      <th>cxpublic.com</th>
      <td>(US)</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>17</td>
      <td>8</td>
      <td>481892</td>
    </tr>
    <tr>
      <th>twimg.com</th>
      <td>(US) Twitter, Inc.</td>
      <td>(US) ANS Communications, Inc</td>
      <td>5</td>
      <td>5.43478</td>
      <td>122</td>
      <td>0.172431</td>
      <td>0</td>
      <td>40</td>
      <td>2412954</td>
    </tr>
    <tr>
      <th>wt-safetag.com</th>
      <td>(DE) Webtrekk GmbH</td>
      <td>(DE) Webtrekk GmbH</td>
      <td>5</td>
      <td>5.43478</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>5</td>
      <td>338</td>
      <td>1145194</td>
    </tr>
    <tr>
      <th>infogram.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>153</td>
      <td>0.216245</td>
      <td>0</td>
      <td>0</td>
      <td>10031759</td>
    </tr>
    <tr>
      <th>unpkg.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>8</td>
      <td>0</td>
      <td>1273714</td>
    </tr>
    <tr>
      <th>adup-tech.com</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>5</td>
      <td>5.43478</td>
      <td>50</td>
      <td>0.0706684</td>
      <td>18</td>
      <td>912</td>
      <td>512538</td>
    </tr>
    <tr>
      <th>clarium.io</th>
      <td>(FR) ClarityAd</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>22</td>
      <td>0.0310941</td>
      <td>5</td>
      <td>2085</td>
      <td>1360</td>
    </tr>
    <tr>
      <th>technical-service.net</th>
      <td>(DE) CBC Cologne Broadcasting Center GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>2</td>
      <td>3774</td>
      <td>1879</td>
    </tr>
    <tr>
      <th>flashtalking.com</th>
      <td>(US) Flashtalking, Inc.</td>
      <td>(US) Squarespace, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>106</td>
      <td>0.149817</td>
      <td>1</td>
      <td>1237</td>
      <td>2817598</td>
    </tr>
    <tr>
      <th>jifo.co</th>
      <td>(LV) INFOGRAM SIA</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>5</td>
      <td>5.43478</td>
      <td>262</td>
      <td>0.370302</td>
      <td>0</td>
      <td>0</td>
      <td>118091311</td>
    </tr>
    <tr>
      <th>contentspread.net</th>
      <td>(DE)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>4</td>
      <td>4.34783</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>0</td>
      <td>0</td>
      <td>522050</td>
    </tr>
    <tr>
      <th>kameleoon.eu</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>4</td>
      <td>4.34783</td>
      <td>112</td>
      <td>0.158297</td>
      <td>37</td>
      <td>1060</td>
      <td>3148663</td>
    </tr>
    <tr>
      <th>yieldlove.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>55</td>
      <td>0.0777352</td>
      <td>11</td>
      <td>40</td>
      <td>8801612</td>
    </tr>
    <tr>
      <th>gscontxt.net</th>
      <td>(US)</td>
      <td>(US) Oracle Corporation</td>
      <td>4</td>
      <td>4.34783</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>3</td>
      <td>430</td>
      <td>4238</td>
    </tr>
    <tr>
      <th>bootstrapcdn.com</th>
      <td>(PA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>11</td>
      <td>0.015547</td>
      <td>3</td>
      <td>6</td>
      <td>435314</td>
    </tr>
    <tr>
      <th>cdn-solution.net</th>
      <td>(DE) Online Solution Int Ltd</td>
      <td>(US) RIPE Network Coordination Centre</td>
      <td>4</td>
      <td>4.34783</td>
      <td>252</td>
      <td>0.356169</td>
      <td>16</td>
      <td>0</td>
      <td>24064433</td>
    </tr>
    <tr>
      <th>stroeerdigitalmedia.de</th>
      <td></td>
      <td>(DE) InterNetX GmbH</td>
      <td>4</td>
      <td>4.34783</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>2</td>
      <td>0</td>
      <td>245</td>
    </tr>
    <tr>
      <th>moatads.com</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>30</td>
      <td>0.042401</td>
      <td>0</td>
      <td>5218</td>
      <td>615304</td>
    </tr>
    <tr>
      <th>fanmatics.com</th>
      <td>(DE)</td>
      <td>(NL) TransIP BV</td>
      <td>4</td>
      <td>4.34783</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>8</td>
      <td>0</td>
      <td>367541</td>
    </tr>
    <tr>
      <th>creativecdn.com</th>
      <td>(PL) RTB House S.A.</td>
      <td>(NL) RTB-HOUSE (DC-AMS)</td>
      <td>4</td>
      <td>4.34783</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>2</td>
      <td>32</td>
      <td>336</td>
    </tr>
    <tr>
      <th>tchibo.de</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>4</td>
      <td>4.34783</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>387</td>
      <td>172</td>
    </tr>
    <tr>
      <th>rp-online.de</th>
      <td></td>
      <td>(DE) circ IT GmbH &amp; Co KG</td>
      <td>4</td>
      <td>4.34783</td>
      <td>21</td>
      <td>0.0296807</td>
      <td>0</td>
      <td>0</td>
      <td>1411033</td>
    </tr>
    <tr>
      <th>cloudfunctions.net</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>4</td>
      <td>4.34783</td>
      <td>64</td>
      <td>0.0904555</td>
      <td>29</td>
      <td>507</td>
      <td>2240</td>
    </tr>
    <tr>
      <th>quantserve.com</th>
      <td>(US) Quantcast</td>
      <td>(GB) Quantcast Ltd.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>0</td>
      <td>1662</td>
      <td>48152</td>
    </tr>
    <tr>
      <th>sqrt-5041.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>3131</td>
      <td>12498</td>
    </tr>
    <tr>
      <th>jquery.com</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>5</td>
      <td>0</td>
      <td>879226</td>
    </tr>
    <tr>
      <th>localhost</th>
      <td></td>
      <td></td>
      <td>4</td>
      <td>4.34783</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>creative-serving.com</th>
      <td>(NL) Platform161 BV</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>400</td>
    </tr>
    <tr>
      <th>retailads.net</th>
      <td>(DE)</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>4</td>
      <td>4.34783</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>19112</td>
    </tr>
    <tr>
      <th>futalis.de</th>
      <td></td>
      <td>(ZZ) APNIC-STUB</td>
      <td>4</td>
      <td>4.34783</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>280</td>
      <td>1400</td>
    </tr>
    <tr>
      <th>zenaps.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>433</td>
      <td>21046</td>
    </tr>
    <tr>
      <th>yumpu.com</th>
      <td>(CH)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>42</td>
      <td>0.0593614</td>
      <td>0</td>
      <td>88</td>
      <td>2163015</td>
    </tr>
    <tr>
      <th>upscore.com</th>
      <td>(US)</td>
      <td>(DE) Asia Pacific Network Information Centre</td>
      <td>4</td>
      <td>4.34783</td>
      <td>69</td>
      <td>0.0975224</td>
      <td>25</td>
      <td>0</td>
      <td>383909</td>
    </tr>
    <tr>
      <th>vlyby.com</th>
      <td>(DE)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>24</td>
      <td>0.0339208</td>
      <td>9</td>
      <td>7</td>
      <td>5753893</td>
    </tr>
    <tr>
      <th>sportbuzzer.de</th>
      <td></td>
      <td>(DE) Verlagsgesellschaft Madsack GmbH &amp; Co.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>24</td>
      <td>0.0339208</td>
      <td>0</td>
      <td>0</td>
      <td>459325</td>
    </tr>
    <tr>
      <th>districtm.io</th>
      <td>(CA) District M Inc.</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>4</td>
      <td>4.34783</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>imgix.net</th>
      <td>(US) Zebrafish Labs</td>
      <td>(US) Fastly</td>
      <td>4</td>
      <td>4.34783</td>
      <td>120</td>
      <td>0.169604</td>
      <td>7</td>
      <td>149</td>
      <td>11872463</td>
    </tr>
    <tr>
      <th>akamaized.net</th>
      <td>(US) Akamai Technologies, inc.</td>
      <td>(DE) Telefonica Germany GmbH &amp; Co. OHG</td>
      <td>3</td>
      <td>3.26087</td>
      <td>11</td>
      <td>0.015547</td>
      <td>2</td>
      <td>239</td>
      <td>23235862</td>
    </tr>
    <tr>
      <th>yimg.com</th>
      <td>(US) Oath Inc.</td>
      <td>(US) Oath Holdings Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>38</td>
      <td>0.053708</td>
      <td>3</td>
      <td>0</td>
      <td>666371</td>
    </tr>
    <tr>
      <th>ablida.net</th>
      <td>(DE)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>3</td>
      <td>17</td>
      <td>11168</td>
    </tr>
    <tr>
      <th>plenigo.com</th>
      <td></td>
      <td>(DE) ProfitBricks Customers Karlsruhe 2</td>
      <td>3</td>
      <td>3.26087</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>3</td>
      <td>0</td>
      <td>583074</td>
    </tr>
    <tr>
      <th>bidr.io</th>
      <td>(FR)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>1665</td>
      <td>0</td>
    </tr>
    <tr>
      <th>c-i.as</th>
      <td></td>
      <td>(DE) Filoo GmbH</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>3</td>
      <td>0</td>
      <td>19494</td>
    </tr>
    <tr>
      <th>ad-srv.net</th>
      <td>(DE)</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>3</td>
      <td>3.26087</td>
      <td>15</td>
      <td>0.0212005</td>
      <td>0</td>
      <td>135</td>
      <td>56963</td>
    </tr>
    <tr>
      <th>sitescout.com</th>
      <td></td>
      <td>(US) Zayo Bandwidth</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>1656</td>
      <td>0</td>
    </tr>
    <tr>
      <th>vtracy.de</th>
      <td></td>
      <td>(DE) myLoc managed IT AG</td>
      <td>3</td>
      <td>3.26087</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>3089</td>
      <td>34512</td>
    </tr>
    <tr>
      <th>tiqcdn.com</th>
      <td>(US)</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>61</td>
      <td>0.0862154</td>
      <td>28</td>
      <td>66</td>
      <td>1854881</td>
    </tr>
    <tr>
      <th>clickagy.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>1812</td>
      <td>0</td>
    </tr>
    <tr>
      <th>rfihub.com</th>
      <td>(US) Zeta Global</td>
      <td>(NL) Sizmek DSP, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>1671</td>
      <td>0</td>
    </tr>
    <tr>
      <th>df-srv.de</th>
      <td></td>
      <td>(DE) Filoo GmbH</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>1</td>
      <td>0</td>
      <td>7820</td>
    </tr>
    <tr>
      <th>_.rocks</th>
      <td></td>
      <td></td>
      <td>3</td>
      <td>3.26087</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ctnsnet.com</th>
      <td>(GB) Crimtan</td>
      <td>(US) Google LLC</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>1803</td>
      <td>0</td>
    </tr>
    <tr>
      <th>resetdigital.co</th>
      <td>(US) Reset Digital</td>
      <td>(US) Google LLC</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>1653</td>
      <td>258</td>
    </tr>
    <tr>
      <th>hotjar.com</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>27</td>
      <td>0.0381609</td>
      <td>0</td>
      <td>0</td>
      <td>1848512</td>
    </tr>
    <tr>
      <th>rlcdn.com</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>2</td>
      <td>1623</td>
      <td>0</td>
    </tr>
    <tr>
      <th>hariken.co</th>
      <td>(BR) Hariken</td>
      <td>(US) Google LLC</td>
      <td>3</td>
      <td>3.26087</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>156</td>
      <td>0</td>
    </tr>
    <tr>
      <th>crazyegg.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>16</td>
      <td>0.0226139</td>
      <td>0</td>
      <td>6</td>
      <td>164408</td>
    </tr>
    <tr>
      <th>asadcdn.com</th>
      <td>(DE)</td>
      <td>(DE) Telefonica Germany GmbH &amp; Co. OHG</td>
      <td>3</td>
      <td>3.26087</td>
      <td>613</td>
      <td>0.866394</td>
      <td>305</td>
      <td>0</td>
      <td>3808455</td>
    </tr>
    <tr>
      <th>npttech.com</th>
      <td>Piano Software</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>3</td>
      <td>0</td>
      <td>53319</td>
    </tr>
    <tr>
      <th>unicef.de</th>
      <td></td>
      <td>(DE) OpenIT GmbH</td>
      <td>3</td>
      <td>3.26087</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>1663</td>
      <td>215</td>
    </tr>
    <tr>
      <th>fbcdn.net</th>
      <td>(US) Facebook, Inc.</td>
      <td>(US) Facebook, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>383</td>
      <td>55562</td>
    </tr>
    <tr>
      <th>truste.com</th>
      <td></td>
      <td>(US) TekTonic</td>
      <td>3</td>
      <td>3.26087</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>270</td>
      <td>84432</td>
    </tr>
    <tr>
      <th>viralize.tv</th>
      <td>(US)</td>
      <td>(US) Rackspace Hosting</td>
      <td>3</td>
      <td>3.26087</td>
      <td>57</td>
      <td>0.080562</td>
      <td>34</td>
      <td>1388</td>
      <td>1804954</td>
    </tr>
    <tr>
      <th>mycleverpush.com</th>
      <td>(PA)</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>3</td>
      <td>3.26087</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>3</td>
      <td>81</td>
      <td>949326</td>
    </tr>
    <tr>
      <th>hs-data.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>58</td>
      <td>0.0819753</td>
      <td>0</td>
      <td>9</td>
      <td>1566184</td>
    </tr>
    <tr>
      <th>krxd.net</th>
      <td>(US) Salesforce.com, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>1481</td>
      <td>0</td>
    </tr>
    <tr>
      <th>brealtime.com</th>
      <td>(US) Engine</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>4</td>
      <td>1638</td>
      <td>8334</td>
    </tr>
    <tr>
      <th>content-garden.com</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>3</td>
      <td>3.26087</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>274721</td>
    </tr>
    <tr>
      <th>kobel.io</th>
      <td>(DE) Axel Springer SE</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>219</td>
      <td>45</td>
    </tr>
    <tr>
      <th>urban-media.com</th>
      <td>(DE)</td>
      <td>(FR) OVH SAS</td>
      <td>3</td>
      <td>3.26087</td>
      <td>22</td>
      <td>0.0310941</td>
      <td>5</td>
      <td>0</td>
      <td>1852492</td>
    </tr>
    <tr>
      <th>loggly.com</th>
      <td>(US) SolarWinds Worldwide, LLC</td>
      <td>(US) SolarWinds, Inc.</td>
      <td>3</td>
      <td>3.26087</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>2</td>
      <td>308</td>
      <td>95</td>
    </tr>
    <tr>
      <th>piano.io</th>
      <td>(US) Piano Software</td>
      <td>(US) Pantheon</td>
      <td>3</td>
      <td>3.26087</td>
      <td>101</td>
      <td>0.14275</td>
      <td>50</td>
      <td>402</td>
      <td>1794240</td>
    </tr>
    <tr>
      <th>mateti.net</th>
      <td></td>
      <td>(DE) Webtrekk GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>25</td>
      <td>0.0353342</td>
      <td>8</td>
      <td>636</td>
      <td>771414</td>
    </tr>
    <tr>
      <th>biallo.de</th>
      <td></td>
      <td>(DE) 1&amp;1 IONOS SE</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>57090</td>
    </tr>
    <tr>
      <th>welect.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>0</td>
      <td>65503</td>
    </tr>
    <tr>
      <th>msgp.pl</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>2</td>
      <td>2.17391</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>2</td>
      <td>0</td>
      <td>3309327</td>
    </tr>
    <tr>
      <th>meinsol.de</th>
      <td></td>
      <td>(DE) Netzindianer sp. z o. o.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>0</td>
      <td>0</td>
      <td>98541</td>
    </tr>
    <tr>
      <th>atdmt.com</th>
      <td>(US) Facebook, Inc.</td>
      <td>(US) MCI Communications Services, Inc. d/b/a Verizon Business</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>86</td>
    </tr>
    <tr>
      <th>atonato.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>2</td>
      <td>0</td>
      <td>42</td>
    </tr>
    <tr>
      <th>wp.com</th>
      <td>(US) Automattic, Inc.</td>
      <td>(US) Automattic, Inc</td>
      <td>2</td>
      <td>2.17391</td>
      <td>57</td>
      <td>0.080562</td>
      <td>21</td>
      <td>175</td>
      <td>1492228</td>
    </tr>
    <tr>
      <th>biallo3.de</th>
      <td></td>
      <td>(DE) 1&amp;1 IONOS SE</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>26</td>
      <td>8067</td>
    </tr>
    <tr>
      <th>omnitagjs.com</th>
      <td>(FR) Omnitag JS</td>
      <td>(FR) Iguane Solutions Technical Team</td>
      <td>2</td>
      <td>2.17391</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>608</td>
      <td>2057</td>
    </tr>
    <tr>
      <th>getback.ch</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>33</td>
      <td>0.0466411</td>
      <td>7</td>
      <td>264</td>
      <td>760372</td>
    </tr>
    <tr>
      <th>cmcdn.de</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>0</td>
      <td>0</td>
      <td>201506</td>
    </tr>
    <tr>
      <th>tiktok.com</th>
      <td>(KY) TIKTOK LTD</td>
      <td>(SG) Asia Pacific Network Information Centre</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>189844</td>
    </tr>
    <tr>
      <th>addthisedge.com</th>
      <td>(US)</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>5114</td>
    </tr>
    <tr>
      <th>addthis.com</th>
      <td></td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>2</td>
      <td>0</td>
      <td>778767</td>
    </tr>
    <tr>
      <th>fontawesome.com</th>
      <td>(US)</td>
      <td>(US) StackPath, LLC.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>1</td>
      <td>0</td>
      <td>290737</td>
    </tr>
    <tr>
      <th>ligatus.com</th>
      <td></td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
    </tr>
    <tr>
      <th>wordlift.io</th>
      <td>(IT) InsideOut10</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>626840</td>
    </tr>
    <tr>
      <th>awin.com</th>
      <td>(US)</td>
      <td>(GB) AWIN LTD</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>94680</td>
    </tr>
    <tr>
      <th>onesignal.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>3</td>
      <td>26</td>
      <td>902090</td>
    </tr>
    <tr>
      <th>googleoptimize.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>2</td>
      <td>2.17391</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>3</td>
      <td>26</td>
      <td>517348</td>
    </tr>
    <tr>
      <th>scdn.co</th>
      <td>(SE) SPOTIFY AB</td>
      <td>(US) Fastly</td>
      <td>2</td>
      <td>2.17391</td>
      <td>52</td>
      <td>0.0734951</td>
      <td>0</td>
      <td>0</td>
      <td>4075139</td>
    </tr>
    <tr>
      <th>adcell.com</th>
      <td>(DE)</td>
      <td>(DE) Soprado GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>0</td>
      <td>21975</td>
    </tr>
    <tr>
      <th>spotify.com</th>
      <td>(SE) Spotify AB</td>
      <td>(US) Google LLC</td>
      <td>2</td>
      <td>2.17391</td>
      <td>16</td>
      <td>0.0226139</td>
      <td>0</td>
      <td>78</td>
      <td>262594</td>
    </tr>
    <tr>
      <th>youtube-nocookie.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>2</td>
      <td>2.17391</td>
      <td>62</td>
      <td>0.0876288</td>
      <td>1</td>
      <td>98</td>
      <td>14150252</td>
    </tr>
    <tr>
      <th>meine-vrm.de</th>
      <td></td>
      <td>(DE) evolver services GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>227</td>
      <td>0.320834</td>
      <td>8</td>
      <td>0</td>
      <td>4152880</td>
    </tr>
    <tr>
      <th>allgemeine-zeitung.de</th>
      <td></td>
      <td>(DE) evolver services GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>0</td>
      <td>0</td>
      <td>727776</td>
    </tr>
    <tr>
      <th>nexx.cloud</th>
      <td>(DE) 3Q GmbH</td>
      <td>(US) Microsoft Corporation</td>
      <td>2</td>
      <td>2.17391</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>0</td>
      <td>0</td>
      <td>7976388</td>
    </tr>
    <tr>
      <th>bannersnack.com</th>
      <td>(US) Smarketer LLC.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>24</td>
      <td>0.0339208</td>
      <td>0</td>
      <td>58</td>
      <td>648234</td>
    </tr>
    <tr>
      <th>icony.com</th>
      <td>(DE) ICONY GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>694</td>
      <td>26364</td>
    </tr>
    <tr>
      <th>tiktokcdn.com</th>
      <td>(KY) TIKTOK LTD</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>2</td>
      <td>0</td>
      <td>55592</td>
    </tr>
    <tr>
      <th>richaudience.com</th>
      <td>(ES)</td>
      <td>(ES) Red de CanalPyme</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>1257</td>
      <td>7318</td>
    </tr>
    <tr>
      <th>uri.sh</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>1033042</td>
    </tr>
    <tr>
      <th>disquscdn.com</th>
      <td>(US) Disqus, Inc.</td>
      <td>(US) Fastly</td>
      <td>2</td>
      <td>2.17391</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>8</td>
      <td>0</td>
      <td>2427487</td>
    </tr>
    <tr>
      <th>kaspersky.com</th>
      <td>(RU) AO Kaspersky Lab</td>
      <td>(RU) Kaspersky Lab AO</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>44237</td>
    </tr>
    <tr>
      <th>ravenjs.com</th>
      <td>(US)</td>
      <td>(US) GitHub, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>2</td>
      <td>0</td>
      <td>75510</td>
    </tr>
    <tr>
      <th>h-cdn.com</th>
      <td>(IL) Hola Networks Ltd.</td>
      <td>(US) Fastly</td>
      <td>2</td>
      <td>2.17391</td>
      <td>32</td>
      <td>0.0452278</td>
      <td>6</td>
      <td>186</td>
      <td>4860962</td>
    </tr>
    <tr>
      <th>welt.de</th>
      <td></td>
      <td>(DE) Boreus Rechenzentrum GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>6</td>
      <td>34</td>
      <td>580712</td>
    </tr>
    <tr>
      <th>onaudience.com</th>
      <td>(UK)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>140</td>
      <td>0</td>
    </tr>
    <tr>
      <th>s-i-r.de</th>
      <td></td>
      <td>(DE) Stuttgart Internet Regional GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>19</td>
      <td>0.026854</td>
      <td>6</td>
      <td>451</td>
      <td>411410</td>
    </tr>
    <tr>
      <th>licdn.com</th>
      <td>(US) LinkedIn Corporation</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>12966</td>
    </tr>
    <tr>
      <th>ads-twitter.com</th>
      <td>(US) Twitter, Inc.</td>
      <td>(US) Fastly</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>20640</td>
    </tr>
    <tr>
      <th>linkedin.com</th>
      <td>(US) LinkedIn Corporation</td>
      <td>(US) Microsoft Corporation</td>
      <td>2</td>
      <td>2.17391</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>2</td>
      <td>183</td>
      <td>0</td>
    </tr>
    <tr>
      <th>stellenanzeigen.de</th>
      <td></td>
      <td>(DE) stellenanzeigen.de GmbH &amp; Co. KG</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>221790</td>
    </tr>
    <tr>
      <th>t.co</th>
      <td>(US) Twitter, Inc.</td>
      <td>(US) Twitter Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>473</td>
      <td>172</td>
    </tr>
    <tr>
      <th>s-onetag.com</th>
      <td>(US) sovrn Holdings, Inc</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>4</td>
      <td>0</td>
      <td>262201</td>
    </tr>
    <tr>
      <th>netpoint-media.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>2</td>
      <td>2.17391</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>9</td>
      <td>553</td>
      <td>332526</td>
    </tr>
    <tr>
      <th>vi-serve.com</th>
      <td>(GB)</td>
      <td>(US) Highwinds Network Group, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>24</td>
      <td>0.0339208</td>
      <td>24</td>
      <td>938</td>
      <td>641092</td>
    </tr>
    <tr>
      <th>inforsea.com</th>
      <td>(GB)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>18</td>
      <td>0</td>
      <td>1173768</td>
    </tr>
    <tr>
      <th>ln-online.de</th>
      <td></td>
      <td>(DE) Verlagsgesellschaft Madsack GmbH &amp; Co.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>27697</td>
    </tr>
    <tr>
      <th>dwin1.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>50741</td>
    </tr>
    <tr>
      <th>rvty.net</th>
      <td>(DE)</td>
      <td>(DE) myLoc managed IT AG</td>
      <td>2</td>
      <td>2.17391</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>14</td>
      <td>97889</td>
    </tr>
    <tr>
      <th>prmutv.co</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>2</td>
      <td>2.17391</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>74</td>
      <td>92</td>
    </tr>
    <tr>
      <th>ibytedtos.com</th>
      <td>(KY) Lemon Inc</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>4</td>
      <td>44</td>
      <td>190016</td>
    </tr>
    <tr>
      <th>onthe.io</th>
      <td>(GB)</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>29</td>
      <td>0.0409877</td>
      <td>16</td>
      <td>461</td>
      <td>218906</td>
    </tr>
    <tr>
      <th>flourish.studio</th>
      <td>(FR) Kiln Enterprises Ltd</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>0</td>
      <td>32450</td>
    </tr>
    <tr>
      <th>permutive.com</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>2</td>
      <td>2.17391</td>
      <td>44</td>
      <td>0.0621882</td>
      <td>0</td>
      <td>110</td>
      <td>837775</td>
    </tr>
    <tr>
      <th>trustarc.com</th>
      <td>(US) TrustArc Inc.</td>
      <td>(US) TekTonic</td>
      <td>2</td>
      <td>2.17391</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>4</td>
      <td>48</td>
      <td>94878</td>
    </tr>
    <tr>
      <th>bitmovin.com</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>92</td>
    </tr>
    <tr>
      <th>noz-cdn.de</th>
      <td></td>
      <td>(DE) Boreus Rechenzentrum GmbH</td>
      <td>2</td>
      <td>2.17391</td>
      <td>146</td>
      <td>0.206352</td>
      <td>21</td>
      <td>0</td>
      <td>6541572</td>
    </tr>
    <tr>
      <th>createjs.com</th>
      <td>(PA)</td>
      <td>(US) Media Temple, Inc.</td>
      <td>2</td>
      <td>2.17391</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>623317</td>
    </tr>
    <tr>
      <th>pingdom.net</th>
      <td>(SE) Pingdom AB</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>341</td>
      <td>6294</td>
    </tr>
    <tr>
      <th>app.link</th>
      <td>(US) Branch</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>109</td>
      <td>180</td>
    </tr>
    <tr>
      <th>vxcp.de</th>
      <td></td>
      <td>(DE) i12 GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>0</td>
      <td>496505</td>
    </tr>
    <tr>
      <th>wlct-one.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>11</td>
      <td>45272</td>
    </tr>
    <tr>
      <th>main-echo-cdn.de</th>
      <td></td>
      <td>(DE) diva-e Datacenters GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>157</td>
      <td>0.221899</td>
      <td>34</td>
      <td>17</td>
      <td>7867878</td>
    </tr>
    <tr>
      <th>rnd.de</th>
      <td></td>
      <td>(DE) Verlagsgesellschaft Madsack GmbH &amp; Co.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>99</td>
      <td>0.139923</td>
      <td>0</td>
      <td>0</td>
      <td>7020483</td>
    </tr>
    <tr>
      <th>cookiepro.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>8</td>
      <td>0</td>
      <td>1749103</td>
    </tr>
    <tr>
      <th>abtasty.com</th>
      <td>(FR) Liwio</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>2</td>
      <td>0</td>
      <td>335350</td>
    </tr>
    <tr>
      <th>akstat.io</th>
      <td>(CA)</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>intellitxt.com</th>
      <td>(GB) Vibrant Media Limited</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>563</td>
      <td>4158</td>
    </tr>
    <tr>
      <th>branch.io</th>
      <td>(US) Branch</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>0</td>
      <td>161575</td>
    </tr>
    <tr>
      <th>go-mpulse.net</th>
      <td>(US) Akamai Technologies, inc.</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>324</td>
      <td>415578</td>
    </tr>
    <tr>
      <th>googleusercontent.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>141288</td>
    </tr>
    <tr>
      <th>emsservice.de</th>
      <td></td>
      <td>(EU) Akamai Technologies</td>
      <td>1</td>
      <td>1.08696</td>
      <td>39</td>
      <td>0.0551213</td>
      <td>0</td>
      <td>0</td>
      <td>291022</td>
    </tr>
    <tr>
      <th>kaltura.com</th>
      <td>(US) Kaltura Inc</td>
      <td>(US) Kaltura Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>1758993</td>
    </tr>
    <tr>
      <th>dymatrix.cloud</th>
      <td>(DE) Dymatrix Consulting Group GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>246</td>
    </tr>
    <tr>
      <th>ix.de</th>
      <td></td>
      <td>(DE) Heise Gruppe GmbH &amp; Co. KG</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>7980</td>
    </tr>
    <tr>
      <th>lr-digital.de</th>
      <td></td>
      <td>(FR) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>1</td>
      <td>0</td>
      <td>285286</td>
    </tr>
    <tr>
      <th>cloudimg.io</th>
      <td>(FR) REFLUENCE</td>
      <td>(FR) virtualisation</td>
      <td>1</td>
      <td>1.08696</td>
      <td>156</td>
      <td>0.220485</td>
      <td>8</td>
      <td>0</td>
      <td>1518568</td>
    </tr>
    <tr>
      <th>bluesummit.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>14804</td>
    </tr>
    <tr>
      <th>igstatic.com</th>
      <td>(FR) i-graal</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>17584</td>
    </tr>
    <tr>
      <th>aspnetcdn.com</th>
      <td>(US) Microsoft Corporation</td>
      <td>(US) ANS Communications, Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>42136</td>
    </tr>
    <tr>
      <th>offerista.com</th>
      <td>(DE)</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>0</td>
      <td>599910</td>
    </tr>
    <tr>
      <th>stackpathdns.com</th>
      <td>(US) NetDNA, LLC.</td>
      <td>(US) StackPath, LLC.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>6</td>
      <td>15560</td>
    </tr>
    <tr>
      <th>intercom.io</th>
      <td>(IE) Intercom Ops</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>22928</td>
    </tr>
    <tr>
      <th>noz.de</th>
      <td></td>
      <td>(DE) Boreus Rechenzentrum GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>0</td>
      <td>0</td>
      <td>797776</td>
    </tr>
    <tr>
      <th>shz.de</th>
      <td></td>
      <td>(DE) Boreus Rechenzentrum GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>0</td>
      <td>0</td>
      <td>71503</td>
    </tr>
    <tr>
      <th>icony-hosting.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>14</td>
      <td>0.0197871</td>
      <td>0</td>
      <td>0</td>
      <td>40853</td>
    </tr>
    <tr>
      <th>intercomcdn.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>0</td>
      <td>531004</td>
    </tr>
    <tr>
      <th>ovb24.de</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>0</td>
      <td>76152</td>
    </tr>
    <tr>
      <th>hotjar.io</th>
      <td>(MT) Hotjar Ltd</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>selfcampaign.com</th>
      <td>(DE) B2B Media Group EMEA GmbH</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>422</td>
      <td>43</td>
    </tr>
    <tr>
      <th>3qsdn.com</th>
      <td>(DE)</td>
      <td>(DE) 3Q Medien GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>0</td>
      <td>810425</td>
    </tr>
    <tr>
      <th>marktjagd.de</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>15</td>
      <td>0.0212005</td>
      <td>0</td>
      <td>0</td>
      <td>72721</td>
    </tr>
    <tr>
      <th>motoso.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>49172</td>
    </tr>
    <tr>
      <th>wcfbc.net</th>
      <td></td>
      <td>(DE) Webtrekk GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>85</td>
      <td>138</td>
    </tr>
    <tr>
      <th>sparwelt.click</th>
      <td>(DE) SPARWELT GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>1050862</td>
    </tr>
    <tr>
      <th>marktjagd.com</th>
      <td>(DE)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>48</td>
      <td>0.0678416</td>
      <td>0</td>
      <td>0</td>
      <td>365986</td>
    </tr>
    <tr>
      <th>adspirit.de</th>
      <td></td>
      <td>(DE) Host Europe GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>0</td>
      <td>177832</td>
    </tr>
    <tr>
      <th>adlooxtracking.com</th>
      <td>(FR) Adloox</td>
      <td>(FR) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>117</td>
      <td>86783</td>
    </tr>
    <tr>
      <th>jobs-im-suedwesten.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>12</td>
      <td>30734</td>
    </tr>
    <tr>
      <th>stuttgarter-zeitung.de</th>
      <td></td>
      <td>(DE) Boreus Rechenzentrum GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>0</td>
      <td>109330</td>
    </tr>
    <tr>
      <th>oberpfalzmedien.de</th>
      <td></td>
      <td>(DE) Der neue Tag Oberpfaelzischer Kurier Druck- und Verlagshaus GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>366</td>
      <td>123960</td>
    </tr>
    <tr>
      <th>heilbronnerstimme.de</th>
      <td></td>
      <td>(DE) fidion GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>95</td>
      <td>0.13427</td>
      <td>25</td>
      <td>17</td>
      <td>3770129</td>
    </tr>
    <tr>
      <th>conrad.de</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>71</td>
      <td>0</td>
    </tr>
    <tr>
      <th>mediamathtag.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>0</td>
      <td>307</td>
      <td>124781</td>
    </tr>
    <tr>
      <th>conrad.com</th>
      <td>(DE) Conrad Electronic SE</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>9</td>
      <td>20917</td>
    </tr>
    <tr>
      <th>unbounce.com</th>
      <td>(PA)</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>13</td>
      <td>120361</td>
    </tr>
    <tr>
      <th>alexametrics.com</th>
      <td>(US) Alexa Internet</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>450</td>
      <td>43</td>
    </tr>
    <tr>
      <th>omsnative.de</th>
      <td></td>
      <td></td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>pinpoll.com</th>
      <td></td>
      <td>(US) Microsoft Corporation</td>
      <td>1</td>
      <td>1.08696</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>2</td>
      <td>0</td>
      <td>100822</td>
    </tr>
    <tr>
      <th>omny.fm</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>0</td>
      <td>632507</td>
    </tr>
    <tr>
      <th>scene7.com</th>
      <td>(US) Adobe Inc.</td>
      <td>(IN) Asia Pacific Network Information Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>51</td>
      <td>67965</td>
    </tr>
    <tr>
      <th>omnycontent.com</th>
      <td>(US) The E. W. Scripps Company</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>20</td>
      <td>138086</td>
    </tr>
    <tr>
      <th>s-p-m.ch</th>
      <td></td>
      <td>(CH) Beja Group GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>19734</td>
    </tr>
    <tr>
      <th>commander1.com</th>
      <td>(FR) Fjord Technologies</td>
      <td>(FR) FJORD TECHNOLOGIES</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>13</td>
      <td>86</td>
    </tr>
    <tr>
      <th>lkqd.net</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>324</td>
      <td>22</td>
    </tr>
    <tr>
      <th>trustcommander.net</th>
      <td>(FR) Fjord Technologies</td>
      <td>(US) MCI Communications Services, Inc. d/b/a Verizon Business</td>
      <td>1</td>
      <td>1.08696</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>2</td>
      <td>0</td>
      <td>633205</td>
    </tr>
    <tr>
      <th>windows.net</th>
      <td>(US) Microsoft Corporation</td>
      <td>(US) Microsoft Corporation</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>309732</td>
    </tr>
    <tr>
      <th>slgnt.eu</th>
      <td></td>
      <td>(US) Tiggee LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>1242</td>
    </tr>
    <tr>
      <th>rackcdn.com</th>
      <td>(US) Rackspace US, Inc.</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>0</td>
      <td>15160</td>
    </tr>
    <tr>
      <th>kaloo.ga</th>
      <td></td>
      <td></td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>contentinsights.com</th>
      <td>(BG) Content Insights</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>6</td>
      <td>290</td>
      <td>0</td>
    </tr>
    <tr>
      <th>omniv.io</th>
      <td>(DE) DDV Mediengruppe GmbH &amp; Co. KG</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>15488</td>
    </tr>
    <tr>
      <th>artikelscore.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>4</td>
      <td>354</td>
      <td>529635</td>
    </tr>
    <tr>
      <th>rawr.at</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>11</td>
      <td>0.015547</td>
      <td>11</td>
      <td>0</td>
      <td>426</td>
    </tr>
    <tr>
      <th>batch.com</th>
      <td>(CA)</td>
      <td>(FR) Dedicated Servers</td>
      <td>1</td>
      <td>1.08696</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>4</td>
      <td>0</td>
      <td>227720</td>
    </tr>
    <tr>
      <th>mannheimer-morgen.de</th>
      <td></td>
      <td>(DE) Newsfactory GmbH - housing</td>
      <td>1</td>
      <td>1.08696</td>
      <td>139</td>
      <td>0.196458</td>
      <td>0</td>
      <td>0</td>
      <td>5136170</td>
    </tr>
    <tr>
      <th>myfonts.net</th>
      <td>(US) MyFonts Inc.</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>eon.de</th>
      <td></td>
      <td>(DE) Adacor Hosting GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>93</td>
      <td>0</td>
    </tr>
    <tr>
      <th>audiencemanager.de</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>5</td>
      <td>178</td>
      <td>278218</td>
    </tr>
    <tr>
      <th>tickaroo.com</th>
      <td>(US)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>34</td>
      <td>0.0480545</td>
      <td>34</td>
      <td>0</td>
      <td>444243</td>
    </tr>
    <tr>
      <th>yoochoose.net</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>imrworldwide.com</th>
      <td>(US) The Nielsen Company</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>893</td>
      <td>44</td>
    </tr>
    <tr>
      <th>adalliance.io</th>
      <td>(DE) G+J Electronic Media Sales GmbH</td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>0</td>
      <td>162</td>
      <td>22358</td>
    </tr>
    <tr>
      <th>hscta.net</th>
      <td>(US) HUBSPOT INC.</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>18482</td>
    </tr>
    <tr>
      <th>consentric.de</th>
      <td></td>
      <td>(GB) Microsoft Limited</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>88</td>
      <td>43</td>
    </tr>
    <tr>
      <th>pubmine.com</th>
      <td>(US) Automattic, Inc.</td>
      <td>(IE) Amazon Web Services, Elastic Compute Cloud, EC2, EU</td>
      <td>1</td>
      <td>1.08696</td>
      <td>14</td>
      <td>0.0197871</td>
      <td>0</td>
      <td>56</td>
      <td>399097</td>
    </tr>
    <tr>
      <th>iqdigital.de</th>
      <td></td>
      <td>(DE) Mittwald CM Service GmbH und Co.KG</td>
      <td>1</td>
      <td>1.08696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>3</td>
      <td>22</td>
      <td>225161</td>
    </tr>
    <tr>
      <th>typekit.net</th>
      <td>(US) Adobe Inc.</td>
      <td>(US) Oracle Corporation</td>
      <td>1</td>
      <td>1.08696</td>
      <td>22</td>
      <td>0.0310941</td>
      <td>0</td>
      <td>77</td>
      <td>594754</td>
    </tr>
    <tr>
      <th>ebayadservices.com</th>
      <td>(US) eBay Inc.</td>
      <td>(US) eBay, Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>96</td>
      <td>43</td>
    </tr>
    <tr>
      <th>ebaystatic.com</th>
      <td>(CH) eBay Marketplaces GmbH</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>43</td>
    </tr>
    <tr>
      <th>pushengage.com</th>
      <td>(PA)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>166936</td>
    </tr>
    <tr>
      <th>paypal.com</th>
      <td>(US) PayPal Inc.</td>
      <td>(US) PayPal, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>1492</td>
    </tr>
    <tr>
      <th>rumble.com</th>
      <td>(CA)</td>
      <td>(US) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>0</td>
      <td>50</td>
      <td>261556</td>
    </tr>
    <tr>
      <th>paypalobjects.com</th>
      <td>(US) PayPal Inc.</td>
      <td>(US) PayPal, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>1492</td>
    </tr>
    <tr>
      <th>rmbl.ws</th>
      <td></td>
      <td>(US) SoftLayer Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>13</td>
      <td>0.0183738</td>
      <td>0</td>
      <td>4</td>
      <td>13868249</td>
    </tr>
    <tr>
      <th>wordpress.com</th>
      <td>(US) Automattic, Inc.</td>
      <td>(US) Automattic, Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>74</td>
      <td>0.104589</td>
      <td>7</td>
      <td>13</td>
      <td>349085</td>
    </tr>
    <tr>
      <th>gravatar.com</th>
      <td>(US) Automattic, Inc.</td>
      <td>(US) Automattic, Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>19</td>
      <td>0.026854</td>
      <td>5</td>
      <td>10</td>
      <td>78989</td>
    </tr>
    <tr>
      <th>voltairenet.org</th>
      <td>(FR)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>9</td>
      <td>0.0127203</td>
      <td>9</td>
      <td>0</td>
      <td>243547</td>
    </tr>
    <tr>
      <th>goo.gl</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>dumontnet.de</th>
      <td></td>
      <td>(DE) united-domains AG</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>61</td>
    </tr>
    <tr>
      <th>springer.com</th>
      <td>(NL) Springer Nature B.V.</td>
      <td>(NL) Springer Nature B.V.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>113</td>
      <td>159733</td>
    </tr>
    <tr>
      <th>dumontnext.de</th>
      <td></td>
      <td>(DE) ProfitBricks Management Karlsruhe 8</td>
      <td>1</td>
      <td>1.08696</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>8</td>
      <td>17</td>
      <td>159156</td>
    </tr>
    <tr>
      <th>warenform.de</th>
      <td></td>
      <td>(DE) virtual hosting platforms</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>343</td>
      <td>123758</td>
    </tr>
    <tr>
      <th>architekturzeitung.net</th>
      <td>(DE)</td>
      <td>(DE) Infrastructure</td>
      <td>1</td>
      <td>1.08696</td>
      <td>112</td>
      <td>0.158297</td>
      <td>59</td>
      <td>111</td>
      <td>669454</td>
    </tr>
    <tr>
      <th>addtoany.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>2</td>
      <td>0</td>
      <td>247493</td>
    </tr>
    <tr>
      <th>ethinking.de</th>
      <td></td>
      <td>(DE) Strato AG</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>2</td>
      <td>56086</td>
    </tr>
    <tr>
      <th>yagiay.com</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>272</td>
    </tr>
    <tr>
      <th>aws-cbc.cloud</th>
      <td>(FR) CBC Cologne Broadcasting Center GmbH</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>181</td>
      <td>2</td>
    </tr>
    <tr>
      <th>nmrodam.com</th>
      <td>(US) The Nielsen Company US, LLC</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>1225</td>
      <td>224248</td>
    </tr>
    <tr>
      <th>e-pages.dk</th>
      <td></td>
      <td>(DK) Visiolink</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>56703</td>
    </tr>
    <tr>
      <th>typography.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>brightcove.com</th>
      <td></td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>7</td>
      <td>0.00989357</td>
      <td>0</td>
      <td>581</td>
      <td>9137</td>
    </tr>
    <tr>
      <th>zencdn.net</th>
      <td>(FR) Brightcove, Inc.</td>
      <td>(US) Fastly</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>20751</td>
    </tr>
    <tr>
      <th>brightcove.net</th>
      <td></td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>14</td>
      <td>849718</td>
    </tr>
    <tr>
      <th>wallstreet-online.de</th>
      <td></td>
      <td>(DE) SOPRADO GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>136</td>
      <td>44127</td>
    </tr>
    <tr>
      <th>hubspot.com</th>
      <td>(US) HUBSPOT INC.</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>99</td>
      <td>81433</td>
    </tr>
    <tr>
      <th>finance.si</th>
      <td></td>
      <td>(SI) Posta Slovenije, d.o.o.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>2</td>
      <td>0</td>
      <td>128391</td>
    </tr>
    <tr>
      <th>cookiebot.com</th>
      <td>(DK) CYBOT</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>3</td>
      <td>0</td>
      <td>379611</td>
    </tr>
    <tr>
      <th>theepochtimes.com</th>
      <td>(US)</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>11</td>
      <td>41669</td>
    </tr>
    <tr>
      <th>quantcount.com</th>
      <td>(US) Quantcast</td>
      <td>(US) Internap Holding LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
    </tr>
    <tr>
      <th>hstrck.com</th>
      <td>(DE) HEIMSPIEL Medien GmbH &amp; Co. KG</td>
      <td>(DE) PlusServer GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>11</td>
      <td>86</td>
    </tr>
    <tr>
      <th>fazcdn.net</th>
      <td>(DE) F.A.Z. Electronic Media GmbH</td>
      <td>(US) PSINet, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>12714</td>
    </tr>
    <tr>
      <th>appspot.com</th>
      <td>(US) Google LLC</td>
      <td>(US) Google LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>23818</td>
    </tr>
    <tr>
      <th>hs-edge.net</th>
      <td>(DE) HEIMSPIEL Medien GmbH &amp; Co. KG</td>
      <td>(DE) PlusServer GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>89927</td>
    </tr>
    <tr>
      <th>dotomi.com</th>
      <td>(US) Conversant LLC</td>
      <td>(US) Conversant, LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>492</td>
      <td>0</td>
    </tr>
    <tr>
      <th>bildstatic.de</th>
      <td></td>
      <td>(EU) Akamai Technologies</td>
      <td>1</td>
      <td>1.08696</td>
      <td>40</td>
      <td>0.0565347</td>
      <td>6</td>
      <td>0</td>
      <td>3255291</td>
    </tr>
    <tr>
      <th>rawgit.com</th>
      <td>(US)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>8</td>
      <td>16641</td>
    </tr>
    <tr>
      <th>vodafone.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>24</td>
      <td>0</td>
    </tr>
    <tr>
      <th>usabilla.com</th>
      <td>(NL)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>1</td>
      <td>3</td>
      <td>116406</td>
    </tr>
    <tr>
      <th>igodigital.com</th>
      <td>(US) Salesforce.com, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>203</td>
      <td>17698</td>
    </tr>
    <tr>
      <th>arcgis.com</th>
      <td>(US) ESRI</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>30</td>
      <td>0.042401</td>
      <td>0</td>
      <td>51</td>
      <td>2384232</td>
    </tr>
    <tr>
      <th>githubusercontent.com</th>
      <td>(US) GitHub, Inc.</td>
      <td>(US) GitHub, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>540740</td>
    </tr>
    <tr>
      <th>wrzmty.com</th>
      <td>(US)</td>
      <td>(FR) OVH SAS</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>6</td>
      <td>304</td>
    </tr>
    <tr>
      <th>rtmark.net</th>
      <td></td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>232</td>
      <td>1480</td>
    </tr>
    <tr>
      <th>vhb.de</th>
      <td></td>
      <td>(US) Microsoft Corp</td>
      <td>1</td>
      <td>1.08696</td>
      <td>45</td>
      <td>0.0636015</td>
      <td>16</td>
      <td>362</td>
      <td>1376</td>
    </tr>
    <tr>
      <th>rtclx.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>parsely.com</th>
      <td>(CA)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>0</td>
      <td>618</td>
      <td>102944</td>
    </tr>
    <tr>
      <th>derwesten.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>2430</td>
    </tr>
    <tr>
      <th>nrz.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>2025</td>
    </tr>
    <tr>
      <th>wp.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>1785</td>
    </tr>
    <tr>
      <th>chimpstatic.com</th>
      <td>(US) THE ROCKET SCIENCE GROUP LLC</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>3390</td>
    </tr>
    <tr>
      <th>list-manage.com</th>
      <td>(US) THE ROCKET SCIENCE GROUP LLC</td>
      <td>(US) The Rocket Science Group, LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>127</td>
      <td>6448</td>
    </tr>
    <tr>
      <th>media-amazon.com</th>
      <td>(US) Amazon Technologies, Inc.</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>63840</td>
    </tr>
    <tr>
      <th>ssl-images-amazon.com</th>
      <td>(US) Amazon Technologies, Inc.</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>1873</td>
    </tr>
    <tr>
      <th>leasewebultracdn.com</th>
      <td>(NL)</td>
      <td>(US) Highwinds Network Group, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>3</td>
      <td>0</td>
      <td>5890441</td>
    </tr>
    <tr>
      <th>bundestag.de</th>
      <td></td>
      <td>(DE) Babiel GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>286</td>
      <td>138764</td>
    </tr>
    <tr>
      <th>pnp.de</th>
      <td></td>
      <td>(DE) evolver services GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>15</td>
      <td>0.0212005</td>
      <td>4</td>
      <td>507</td>
      <td>143631</td>
    </tr>
    <tr>
      <th>allesregional.de</th>
      <td></td>
      <td>(DE) SPIEGLHOF media GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>44</td>
      <td>0.0621882</td>
      <td>0</td>
      <td>6</td>
      <td>4573985</td>
    </tr>
    <tr>
      <th>s4p-iapps.com</th>
      <td>(DE)</td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>0</td>
      <td>0</td>
      <td>426390</td>
    </tr>
    <tr>
      <th>freiepresse-display.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>28</td>
      <td>17788</td>
    </tr>
    <tr>
      <th>plyr.io</th>
      <td>(PA)</td>
      <td>(US) Amazon.com, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>2</td>
      <td>0</td>
      <td>11570</td>
    </tr>
    <tr>
      <th>bf-ad.net</th>
      <td>(DE)</td>
      <td>(EU) Akamai Technologies</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>3</td>
      <td>0</td>
      <td>1300521</td>
    </tr>
    <tr>
      <th>finanzen100.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>27</td>
      <td>25328</td>
    </tr>
    <tr>
      <th>bf-tools.net</th>
      <td>(DE)</td>
      <td>(EU) Akamai Technologies</td>
      <td>1</td>
      <td>1.08696</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>4</td>
      <td>0</td>
      <td>49840</td>
    </tr>
    <tr>
      <th>wfxtriggers.com</th>
      <td>(US) TWC Product and Technology, LLC</td>
      <td>(US) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>2</td>
      <td>62</td>
      <td>699</td>
    </tr>
    <tr>
      <th>netdna-ssl.com</th>
      <td>(US) NetDNA, LLC.</td>
      <td>(US) Highwinds Network Group, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>58538</td>
    </tr>
    <tr>
      <th>speedcurve.com</th>
      <td>(US)</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>10</td>
      <td>43450</td>
    </tr>
    <tr>
      <th>brandmetrics.com</th>
      <td>(SE)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>3</td>
      <td>70</td>
      <td>88562</td>
    </tr>
    <tr>
      <th>erne.co</th>
      <td>(SC)</td>
      <td>(BE) OVH BE</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>97</td>
      <td>0</td>
    </tr>
    <tr>
      <th>aminopay.net</th>
      <td>(US) Integral Ad Science, Inc.</td>
      <td>(US) Amazon Technologies Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>94</td>
      <td>86</td>
    </tr>
    <tr>
      <th>bfops.io</th>
      <td>(DE) BurdaForward GmbH</td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>2</td>
      <td>1014</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ga.de</th>
      <td></td>
      <td>(DE) circ IT GmbH &amp; Co KG</td>
      <td>1</td>
      <td>1.08696</td>
      <td>182</td>
      <td>0.257233</td>
      <td>0</td>
      <td>0</td>
      <td>9162252</td>
    </tr>
    <tr>
      <th>conative.de</th>
      <td></td>
      <td>(DE) netcup GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>18</td>
      <td>0.0254406</td>
      <td>0</td>
      <td>14</td>
      <td>965760</td>
    </tr>
    <tr>
      <th>bit.ly</th>
      <td>(US) Bitly</td>
      <td>(US) Bitly Inc</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
    </tr>
    <tr>
      <th>mailchimp.com</th>
      <td></td>
      <td>(US) Akamai Technologies, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>0</td>
      <td>0</td>
      <td>483624</td>
    </tr>
    <tr>
      <th>indivsurvey.de</th>
      <td></td>
      <td>(DE) ZeuSWarE GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>12</td>
      <td>0.0169604</td>
      <td>6</td>
      <td>0</td>
      <td>517294</td>
    </tr>
    <tr>
      <th>wr.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>1720</td>
    </tr>
    <tr>
      <th>fonts.net</th>
      <td>(US) Monotype Imaging Inc</td>
      <td>(US) CenturyLink Communications, LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ikz-online.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>1703</td>
    </tr>
    <tr>
      <th>permutive.app</th>
      <td>(CA)</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>430490</td>
    </tr>
    <tr>
      <th>abendblatt.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>17870</td>
    </tr>
    <tr>
      <th>mpnrs.com</th>
      <td>(DE) M,P,NEWMEDIA, GmbH</td>
      <td>(DE) rh-tec Business GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>63438</td>
    </tr>
    <tr>
      <th>hubspotusercontent20.net</th>
      <td>(US) HUBSPOT INC.</td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>39594</td>
    </tr>
    <tr>
      <th>aachener-zeitung.de</th>
      <td></td>
      <td>(DE) Zeitungsverlag Aachen</td>
      <td>1</td>
      <td>1.08696</td>
      <td>28</td>
      <td>0.0395743</td>
      <td>0</td>
      <td>0</td>
      <td>887369</td>
    </tr>
    <tr>
      <th>dreilaenderschmeck.de</th>
      <td></td>
      <td>(DE) Zeitungsverlag Aachen</td>
      <td>1</td>
      <td>1.08696</td>
      <td>11</td>
      <td>0.015547</td>
      <td>0</td>
      <td>0</td>
      <td>176607</td>
    </tr>
    <tr>
      <th>oecherdeal.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>10</td>
      <td>0.0141337</td>
      <td>0</td>
      <td>0</td>
      <td>141813</td>
    </tr>
    <tr>
      <th>medienhausaachen.de</th>
      <td></td>
      <td>(DE) Zeitungsverlag Aachen</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>3863</td>
    </tr>
    <tr>
      <th>uobsoe.com</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>136</td>
    </tr>
    <tr>
      <th>abendzeitung.de</th>
      <td></td>
      <td>(DE) Medien System Haus internal network</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>195123</td>
    </tr>
    <tr>
      <th>az-muenchen.de</th>
      <td></td>
      <td>(DE) Medien System Haus internal network</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>116836</td>
    </tr>
    <tr>
      <th>congstar.de</th>
      <td></td>
      <td>(DE) T-Systems Multimedia Solution GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>3</td>
      <td>0.0042401</td>
      <td>0</td>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <th>71i.de</th>
      <td></td>
      <td>(DE) ProSiebenSat.1 Tech Solutions GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>5</td>
      <td>0.00706684</td>
      <td>0</td>
      <td>0</td>
      <td>911057</td>
    </tr>
    <tr>
      <th>rta-design.de</th>
      <td></td>
      <td>(ZZ) APNIC-STUB</td>
      <td>1</td>
      <td>1.08696</td>
      <td>4</td>
      <td>0.00565347</td>
      <td>2</td>
      <td>0</td>
      <td>52184</td>
    </tr>
    <tr>
      <th>mgaz.de</th>
      <td></td>
      <td>(DE) Infrastructure</td>
      <td>1</td>
      <td>1.08696</td>
      <td>8</td>
      <td>0.0113069</td>
      <td>0</td>
      <td>0</td>
      <td>136385</td>
    </tr>
    <tr>
      <th>trauer-im-allgaeu.de</th>
      <td></td>
      <td>(DE) ProfitBricks Customers Karlsruhe 2</td>
      <td>1</td>
      <td>1.08696</td>
      <td>17</td>
      <td>0.0240272</td>
      <td>0</td>
      <td>0</td>
      <td>1647144</td>
    </tr>
    <tr>
      <th>wbtrk.net</th>
      <td>(DE) Webtrekk GmbH</td>
      <td>(DE) Webtrekk GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>1</td>
      <td>0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>peiq.de</th>
      <td></td>
      <td>(DE) RIPE Network Coordination Centre</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>bottalk.io</th>
      <td>(US)</td>
      <td>(US) DigitalOcean, LLC</td>
      <td>1</td>
      <td>1.08696</td>
      <td>6</td>
      <td>0.00848021</td>
      <td>6</td>
      <td>0</td>
      <td>1345182</td>
    </tr>
    <tr>
      <th>b-cdn.net</th>
      <td>(US)</td>
      <td>(DE) CDN77 Frankfurt - Bunny CDN</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>3</td>
      <td>10608</td>
    </tr>
    <tr>
      <th>fupa.net</th>
      <td>(US)</td>
      <td>(DE) nbsp GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>25</td>
      <td>0.0353342</td>
      <td>0</td>
      <td>0</td>
      <td>244240</td>
    </tr>
    <tr>
      <th>berliner-zeitung.de</th>
      <td></td>
      <td>(US) Cloudflare, Inc.</td>
      <td>1</td>
      <td>1.08696</td>
      <td>1</td>
      <td>0.00141337</td>
      <td>0</td>
      <td>0</td>
      <td>13962</td>
    </tr>
    <tr>
      <th>wz-media.de</th>
      <td></td>
      <td>(DE) Hetzner Online GmbH</td>
      <td>1</td>
      <td>1.08696</td>
      <td>2</td>
      <td>0.00282674</td>
      <td>0</td>
      <td>0</td>
      <td>107023</td>
    </tr>
  </tbody>
</table><script type="text/javascript">
        jQuery("#third-parties-table").DataTable({"order": [3, "desc"], "className": "compact", "paging": true});
    </script><div class="table-description"><ul><li><b>registrant</b>: The registrant country and name - the organisation that registered the domain name</li><li><b>network</b>: The country and name of the network operator of the IP</li><li><b>websites</b>: Number of websites where this network was used</li><li><b>websites %</b>: Percentage of all visited websites where this network was used</li><li><b>requests</b>: Percentage of network requests this server received during all visits</li><li><b>article_referer</b>: The number requests that contained the URL of a visited article as 'Referer' header</li><li><b>bytes_sent</b>: The number of bytes sent to this server during all visits (via query parameters and POST data)</li><li><b>bytes_received</b>: The number of bytes received from this server during all visits</li></ul></div>


### the complete matrix

The tables provide some overview and the matrix below provides us with the actual facts of *who used whom*. You can hover over the heatmap to see the names. X is website, Y is third party. 

I'd like to leave the reader alone now in her/his attempts to detect structure and find understanding in all this.





<div>                            <div id="99ca9df5-c06c-4eb3-9a6d-3f3f0a45f28b" class="plotly-graph-div" style="height:8000px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("99ca9df5-c06c-4eb3-9a6d-3f3f0a45f28b")) {                    Plotly.newPlot(                        "99ca9df5-c06c-4eb3-9a6d-3f3f0a45f28b",                        [{"coloraxis": "coloraxis", "hovertemplate": "visited page: %{x}<br>third-party request: %{y}<br>percent: %{z}<extra></extra>", "name": "0", "type": "heatmap", "x": ["www.aachener-nachrichten.de", "www.abendzeitung-muenchen.de", "www.aerztezeitung.de", "www.all-in.de", "www.architekturzeitung.com", "www.augsburger-allgemeine.de", "www.azonline.de", "www.badische-zeitung.de", "www.bayernkurier.de", "www.berliner-kurier.de", "www.berliner-zeitung.de", "www.bild.de", "bnn.de", "www.boersen-zeitung.de", "www.bz-berlin.de", "www.das-parlament.de", "deutsche-wirtschafts-nachrichten.de", "www.die-tagespost.de", "www.donaukurier.de", "www.echo-online.de", "www.epochtimes.de", "www.faz.net", "www.focus.de", "www.fr.de", "www.freiepresse.de", "www.freitag.de", "www.general-anzeiger-bonn.de", "www.generalanzeiger.de", "www.goettinger-tageblatt.de", "www.handelsblatt.com", "www.heise.de", "www.idowa.de", "www.infranken.de", "www.juedische-allgemeine.de", "jungefreiheit.de", "www.jungewelt.de", "www.kath.net", "www.kn-online.de", "www.kreiszeitung.de", "linkezeitung.de", "www.ln-online.de", "www.lr-online.de", "www.lvz.de", "www.main-echo.de", "www.mainpost.de", "www.maz-online.de", "www.medical-tribune.de", "www.merkur.de", "www.mittelbayerische.de", "www.morgenweb.de", "www.moz.de", "www.muensterschezeitung.de", "www.mz-web.de", "www.neuepresse.de", "www.neues-deutschland.de", "www.nordbayern.de", "www.nordkurier.de", "www.noz.de", "www.nw.de", "www.nwzonline.de", "www.onetz.de", "www.ostsee-zeitung.de", "www.ovb-online.de", "www.rhein-zeitung.de", "www.rheinpfalz.de", "www.rnz.de", "www.rp-online.de", "www.saarbruecker-zeitung.de", "www.saechsische.de", "www.schwaebische.de", "www.schwarzwaelder-bote.de", "www.sonntagsblatt.de", "spiegel.de", "www.stimme.de", "www.stuttgarter-nachrichten.de", "www.stuttgarter-zeitung.de", "www.sueddeutsche.de", "www.suedkurier.de", "www.svz.de", "www.swp.de", "www.tagesspiegel.de", "taz.de", "www.tz.de", "www.volksfreund.de", "www.volksstimme.de", "www.wa.de", "www.waz.de", "www.welt.de", "www.westfalen-blatt.de", "www.wiesbadener-kurier.de", "www.wz.de", "www.zeit.de"], "xaxis": "x", "y": ["1rx.io", "2mdn.net", "360yield.com", "3lift.com", "3qsdn.com", "71i.de", "_.rocks", "aachener-zeitung.de", "abendblatt.de", "abendzeitung.de", "ablida.net", "abtasty.com", "ad-production-stage.com", "ad-server.eu", "ad-srv.net", "ad.gt", "ad4m.at", "ad4mat.net", "adalliance.io", "adcell.com", "addthis.com", "addthisedge.com", "addtoany.com", "adform.net", "adition.com", "adkernel.com", "adlooxtracking.com", "adnxs.com", "adobedtm.com", "adrtx.net", "ads-twitter.com", "adsafeprotected.com", "adsafety.net", "adscale.de", "adspirit.de", "adsrvr.org", "adup-tech.com", "advertising.com", "agkn.com", "akamaihd.net", "akamaized.net", "akstat.io", "alexametrics.com", "allesregional.de", "allgemeine-zeitung.de", "amazon-adsystem.com", "amazonaws.com", "aminopay.net", "ampproject.org", "aniview.com", "app.link", "appier.net", "appspot.com", "arcgis.com", "architekturzeitung.net", "artefact.com", "artikelscore.de", "asadcdn.com", "aspnetcdn.com", "atdmt.com", "aticdn.net", "atonato.de", "audiencemanager.de", "awin.com", "awin1.com", "aws-cbc.cloud", "az-muenchen.de", "b-cdn.net", "bannersnack.com", "batch.com", "berliner-zeitung.de", "bf-ad.net", "bf-tools.net", "bfops.io", "biallo.de", "biallo3.de", "bidr.io", "bidswitch.net", "bildstatic.de", "bing.com", "bit.ly", "bitmovin.com", "blau.de", "bluekai.com", "bluesummit.de", "boltdns.net", "bootstrapcdn.com", "bottalk.io", "branch.io", "brandmetrics.com", "brealtime.com", "brightcove.com", "brightcove.net", "brillen.de", "bttrack.com", "bundestag.de", "c-i.as", "casalemedia.com", "cdn-solution.net", "cdntrf.com", "chartbeat.com", "chartbeat.net", "cheqzone.com", "chimpstatic.com", "clarium.io", "cleverpush.com", "clickagy.com", "cloudflare.com", "cloudfront.net", "cloudfunctions.net", "cloudimg.io", "cmcdn.de", "commander1.com", "conative.de", "congstar.de", "conrad.com", "conrad.de", "consensu.org", "consentric.de", "content-garden.com", "contentinsights.com", "contentspread.net", "contextweb.com", "cookiebot.com", "cookielaw.org", "cookiepro.com", "crazyegg.com", "createjs.com", "creative-serving.com", "creativecdn.com", "criteo.com", "criteo.net", "crwdcntrl.net", "ctnsnet.com", "cxense.com", "cxpublic.com", "datawrapper.de", "de.com", "demdex.net", "derwesten.de", "df-srv.de", "disqus.com", "disquscdn.com", "districtm.io", "dnacdn.net", "dotomi.com", "doubleclick.net", "doubleverify.com", "dreilaenderschmeck.de", "dspx.tv", "dumontnet.de", "dumontnext.de", "dwcdn.net", "dwin1.com", "dymatrix.cloud", "e-pages.dk", "ebayadservices.com", "ebaystatic.com", "emetriq.de", "emsservice.de", "emxdgt.com", "eon.de", "erne.co", "ethinking.de", "everesttech.net", "exactag.com", "exelator.com", "f11-ads.com", "f11-ads.net", "facebook.com", "facebook.net", "fanmatics.com", "fastly.net", "fazcdn.net", "fbcdn.net", "finance.si", "finanzen100.de", "flashtalking.com", "flourish.studio", "fontawesome.com", "fonts.net", "freiepresse-display.de", "fupa.net", "futalis.de", "ga.de", "geoedge.be", "getback.ch", "ggpht.com", "githubusercontent.com", "glomex.cloud", "glomex.com", "go-mpulse.net", "goo.gl", "google-analytics.com", "google.com", "google.de", "googleadservices.com", "googleapis.com", "googleoptimize.com", "googlesyndication.com", "googletagmanager.com", "googletagservices.com", "googleusercontent.com", "googlevideo.com", "gravatar.com", "gscontxt.net", "gstatic.com", "h-cdn.com", "hariken.co", "haz.de", "heilbronnerstimme.de", "hotjar.com", "hotjar.io", "hs-data.com", "hs-edge.net", "hscta.net", "hstrck.com", "hubspot.com", "hubspotusercontent20.net", "ibillboard.com", "ibytedtos.com", "icony-hosting.de", "icony.com", "id5-sync.com", "idcdn.de", "igodigital.com", "igstatic.com", "ikz-online.de", "imgix.net", "imrworldwide.com", "indexww.com", "indivsurvey.de", "infogram.com", "inforsea.com", "instagram.com", "intellitxt.com", "intercom.io", "intercomcdn.com", "ioam.de", "ippen.space", "iqdigital.de", "ix.de", "jifo.co", "jobs-im-suedwesten.de", "jquery.com", "jsdelivr.net", "justpremium.com", "kaloo.ga", "kaltura.com", "kameleoon.eu", "kaspersky.com", "kobel.io", "krxd.net", "lead-alliance.net", "leasewebultracdn.com", "liadm.com", "licdn.com", "ligatus.com", "lijit.com", "linkedin.com", "list-manage.com", "lkqd.net", "ln-online.de", "localhost", "loggly.com", "lp4.io", "lr-digital.de", "m-t.io", "m6r.eu", "madsack-native.de", "mailchimp.com", "main-echo-cdn.de", "mannheimer-morgen.de", "marktjagd.com", "marktjagd.de", "mateti.net", "mathtag.com", "media-amazon.com", "media01.eu", "medialead.de", "mediamathtag.com", "medienhausaachen.de", "meetrics.net", "meine-vrm.de", "meinsol.de", "mfadsrvr.com", "mgaz.de", "ml314.com", "mlsat02.de", "moatads.com", "mookie1.com", "motoso.de", "mpnrs.com", "msgp.pl", "mxcdn.net", "mycleverpush.com", "myfonts.net", "nativendo.de", "netdna-ssl.com", "netpoint-media.de", "nexx.cloud", "nmrodam.com", "noz-cdn.de", "noz.de", "npttech.com", "nrz.de", "nuggad.net", "o2online.de", "oadts.com", "oberpfalzmedien.de", "oecherdeal.de", "offerista.com", "office-partner.de", "omnitagjs.com", "omniv.io", "omny.fm", "omnycontent.com", "omsnative.de", "omtrdc.net", "onaudience.com", "onesignal.com", "onetag-sys.com", "onetrust.com", "onthe.io", "opecloud.com", "opencmp.net", "openx.net", "opinary.com", "otto.de", "outbrain.com", "outbrainimg.com", "ovb24.de", "parsely.com", "paypal.com", "paypalobjects.com", "peiq.de", "perfectmarket.com", "permutive.app", "permutive.com", "piano.io", "pingdom.net", "pinpoll.com", "plenigo.com", "plista.com", "plyr.io", "pnp.de", "podigee-cdn.net", "podigee.com", "podigee.io", "polyfill.io", "prebid.org", "pressekompass.net", "privacy-mgmt.com", "prmutv.co", "pubmatic.com", "pubmine.com", "purelocalmedia.de", "pushengage.com", "pushwoosh.com", "quantcount.com", "quantserve.com", "rackcdn.com", "ravenjs.com", "rawgit.com", "rawr.at", "recognified.net", "redintelligence.net", "reisereporter.de", "resetdigital.co", "retailads.net", "rfihub.com", "richaudience.com", "rlcdn.com", "rmbl.ws", "rnd.de", "rndtech.de", "rp-online.de", "rqtrk.eu", "rta-design.de", "rtclx.com", "rtmark.net", "rubiconproject.com", "rumble.com", "rvty.net", "s-i-r.de", "s-onetag.com", "s-p-m.ch", "s4p-iapps.com", "sascdn.com", "scdn.co", "scene7.com", "scorecardresearch.com", "selfcampaign.com", "semasio.net", "serving-sys.com", "showheroes.com", "shz.de", "sitescout.com", "slgnt.eu", "smartadserver.com", "smartclip.net", "smartstream.tv", "sonobi.com", "sp-prod.net", "sparwelt.click", "speedcurve.com", "sphere.com", "sportbuzzer.de", "spotify.com", "spotxchange.com", "springer.com", "sqrt-5041.de", "ssl-images-amazon.com", "stackpathdns.com", "stellenanzeigen.de", "stickyadstv.com", "stroeerdigital.de", "stroeerdigitalgroup.de", "stroeerdigitalmedia.de", "stuttgarter-zeitung.de", "t.co", "taboola.com", "tchibo.de", "teads.tv", "technical-service.net", "technoratimedia.com", "telefonica-partner.de", "telekom.de", "theadex.com", "theepochtimes.com", "tickaroo.com", "tiktok.com", "tiktokcdn.com", "tinypass.com", "tiqcdn.com", "transmatico.com", "trauer-im-allgaeu.de", "tremorhub.com", "trmads.eu", "trmcdn.eu", "trustarc.com", "trustcommander.net", "truste.com", "turn.com", "twiago.com", "twimg.com", "twitter.com", "typekit.net", "typography.com", "unbounce.com", "unicef.de", "unpkg.com", "unrulymedia.com", "uobsoe.com", "upscore.com", "urban-media.com", "uri.sh", "usabilla.com", "usercentrics.eu", "userreport.com", "vgwort.de", "vhb.de", "vi-serve.com", "vidazoo.com", "videoreach.com", "viralize.tv", "visx.net", "vlyby.com", "vodafone.de", "voltairenet.org", "vtracy.de", "vxcp.de", "wallstreet-online.de", "warenform.de", "wbtrk.net", "wcfbc.net", "webgains.com", "webgains.io", "weekli.de", "weekli.systems", "welect.de", "welt.de", "wetterkontor.de", "wfxtriggers.com", "windows.net", "wlct-one.de", "wordlift.io", "wordpress.com", "wp.com", "wp.de", "wr.de", "wrzmty.com", "wt-safetag.com", "wz-media.de", "xiti.com", "xplosion.de", "yagiay.com", "yahoo.com", "yieldlab.net", "yieldlove-ad-serving.net", "yieldlove.com", "yieldscale.com", "yimg.com", "yoochoose.net", "youtube-nocookie.com", "youtube.com", "ytimg.com", "yumpu.com", "zemanta.com", "zenaps.com", "zencdn.net", "zeotap.com"], "yaxis": "y", "z": [[null, 2.97, null, null, null, null, null, 0.64, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, 1.83, null, null, 0.3, null, null, null, 1.01, null, 0.24, null, null, null, null, null, 1.02, null, null, null, null, null, null, null, null, 2.47, null, null, null, null, null, null, null, 0.31, null, null, null, null, 0.34, null, 1.01, null, null, 0.25, null, 0.31, null, null, 0.41, null, null, 0.52, 0.48, 0.43, null, null, null, 1.48, null, null, 1.04, 0.35, null, 0.4, 0.16, null, null, null, 1.32, null], [1.59, 0.56, null, null, 0.33, 3.28, null, null, null, null, null, 2.99, 4.81, null, 2.95, null, 11.93, null, 0.13, 1.02, null, 4.16, 2.74, null, null, 0.19, 1.81, null, null, 0.94, 2.89, null, null, null, 0.36, null, null, 1.9, 0.2, null, 1.14, null, 1.11, 5.26, null, null, null, null, null, 6.72, null, 3.03, null, 0.1, null, 2.6, null, null, null, 2.33, 2.36, 0.34, 1.15, null, 5.28, 0.12, null, 0.86, null, null, null, null, 3.16, 4.43, 1.45, null, null, null, 0.09, 0.23, null, 0.6, 1.99, 0.95, 2.54, 0.27, 1.26, 0.2, null, 3.29, 0.17, 0.66], [null, 0.45, null, null, null, null, null, 1.28, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, 0.77, null, 0.19, 0.3, null, null, null, 0.29, null, 0.48, null, null, null, null, null, 0.91, null, null, null, null, null, null, null, null, 0.6, null, null, null, null, 0.57, null, null, 0.62, null, null, null, null, 0.67, null, 0.29, null, null, 0.49, null, 0.31, null, null, 0.83, null, null, 0.26, 0.97, 0.86, null, null, null, 0.79, null, null, 0.52, 0.35, 0.41, 0.8, 0.08, null, null, null, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.73, null, null, null, null, null, null, null, 0.94, 0.87, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.29, null, null, null, 0.94, null, null, null, null, null, null, null, null, null, null, 0.66], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.28, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [3.71, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.32, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 5.59, null, 2.34, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.13, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.83, 6.71, null, 5.3, null, 3.43, null, null, 4.26, null, 2.69, null, null, null, null, null, 2.38, null, 3.53, null, null, null, null, null, 4.63, null, null, null, null, null, null, null, null, null, null, null, null, 3.72, null, null, null, null, null, null, null, 3.89, null, null, 9.85, null, null, 3.74, null, 3.31, null], [null, null, null, null, null, null, 0.11, 0.21, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, 0.1, null, 0.37, 0.08, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, 0.31, null, 0.5, 0.26, 0.22, null, null, 0.22, null, null, 0.31, null, null, 0.57, null, 0.21, null, 0.27, null, 0.1, 0.17, null, 0.29, null, null, 0.12, null, 0.16, null, null, 0.14, null, 0.4, null, null, null, null, 0.29, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.52, null, null, null, null, null, 0.66, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.06, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, 0.1, null, null, 0.08, null, null, null, null, null, 0.24, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, 0.07, null, null, null, null, null, null, null, 0.1, null, null, null, null, 0.17, null, null, null, null, 0.12, null, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, null, null, null, null, null, 0.09, 0.09, null, 0.13, null, null, null, null, null, null], [2.12, null, null, 3.56, null, 3.28, 3.31, null, null, null, null, null, null, null, null, null, null, null, null, 2.17, null, null, null, null, null, null, null, null, null, null, 1.74, null, null, null, null, null, null, null, null, null, null, 4.2, null, null, 3.69, null, null, null, null, null, 2.48, 0.3, 5.75, null, null, null, null, null, null, 2.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.76, null, 1.36, null, null, null, null, null, null, 1.73, null, 2.88, 2.35, 1.99, null], [0.27, null, null, 0.38, null, 0.35, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, 0.5, null, null, 0.43, null, null, null, null, null, 0.31, null, 0.57, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, 0.11, null, null, null, null, null, null, 0.16, null, 0.32, 0.35, 0.17, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.19, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.84, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [3.18, 4.43, 5.42, 1.02, null, 0.94, 3.76, null, null, null, null, null, null, null, 0.09, null, null, null, 0.27, 1.53, null, 4.16, 0.37, 0.19, null, 0.74, 0.08, null, null, null, 1.3, null, null, null, null, null, null, 0.95, 0.61, null, null, 4.54, null, 0.53, 5.64, null, null, 0.15, null, 0.64, 2.79, 0.91, 0.86, null, null, 0.1, 3.01, 2.17, null, 0.51, null, null, 0.29, null, null, null, null, null, null, null, null, null, 1.45, 0.26, null, 0.14, 7.33, 1.17, null, 2.95, null, 1.67, 0.17, 0.09, 3.25, null, 1.02, null, 1.17, 4.58, 1.66, null], [0.13, 0.39, 0.42, 1.02, null, 0.7, 0.22, null, null, null, null, 4.7, null, 12.2, null, null, 17.19, null, null, 0.51, null, null, 1.5, null, null, 0.37, 0.08, null, null, null, null, null, null, null, null, null, null, null, 0.41, null, null, 0.17, null, null, 0.22, null, null, null, null, null, 0.31, 0.15, 0.29, 2.29, null, 0.31, null, null, null, null, 0.34, null, null, 1.05, null, null, 0.61, null, null, null, null, null, 1.05, null, null, null, null, 0.15, null, 2.95, null, null, null, 0.26, 0.2, null, 1.1, 0.2, 0.11, 0.47, 2.15, null], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [1.99, 2.8, 2.5, 2.16, null, 1.64, 2.87, 5.53, null, 6.93, null, 7.47, null, null, 8.68, null, null, null, 3.75, 1.66, 0.84, 1.71, 10.6, 2.6, 3.83, null, 2.87, null, 1.07, 1.09, 0.29, null, 4.8, null, null, null, null, 0.95, 4.17, null, 0.83, 2.02, 0.81, 3.6, 2.82, 0.92, null, 4.19, 4.0, null, 10.85, 2.12, 2.01, 0.76, null, 3.01, 1.62, 1.09, 0.38, 0.81, 7.42, 0.9, null, null, 3.13, 7.78, 0.3, 3.2, null, 0.57, 6.07, null, 1.05, 3.65, 4.36, 5.19, 0.43, 2.79, 0.46, 1.36, 0.31, 6.67, 3.98, 3.89, 5.39, 5.46, 2.59, 5.74, 2.99, 1.41, 1.99, 0.22], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.18, null, null, null, null, null, null, null, 0.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 6.32, null, null, null, 1.29, null, null, null, 0.94, null, null, null, null, null, null, null, null, null, null, 1.33], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, 0.49, 0.5, null, null, 0.74, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.64, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.53, null, null, null, 0.86, null, null, null, 0.63, null, null, null, null, null, null, 0.2, null, null, null, 0.88], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, 0.5, null, null, null, 0.82, 0.66, null, null, null, null, 4.8, null, null, 1.7, null, 9.82, null, null, 1.28, null, 7.33, 1.87, null, 0.48, null, null, null, null, 6.55, null, 1.36, null, null, null, null, null, 1.07, null, null, 0.73, null, 0.71, 1.05, null, null, null, null, null, null, 0.31, 1.52, null, null, null, 0.83, 2.55, 1.76, 8.14, 0.2, 2.36, 1.47, 1.01, 1.05, 0.39, null, null, 0.55, null, null, 0.28, null, null, null, null, 0.14, 6.9, null, 4.26, 0.23, 7.51, null, 1.12, null, 4.37, 1.33, 0.39, 0.99, 0.21, 0.71, null, 6.86], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.44, null, null, null, null, null, null, null, null, null, null, null, 0.52, 2.31, 0.54, 0.95, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.46, null, null, null, null, null, null, null, 0.71, null, null, null, null, null], [1.59, null, 8.33, 2.29, 0.65, 2.11, 4.31, null, null, null, null, null, null, null, null, null, null, null, null, 4.21, null, null, null, null, 0.96, null, null, null, null, null, null, 5.44, null, null, null, null, null, 1.07, null, null, 0.42, 3.19, null, 0.53, 3.04, 0.46, null, null, null, null, 2.79, 3.79, 1.72, 0.57, null, 0.42, null, null, 1.14, 2.64, 1.35, 0.68, null, 5.24, 0.78, null, null, null, null, null, 1.1, null, null, 2.21, 0.48, 0.43, null, 2.94, null, 1.82, null, 3.58, null, null, 3.15, null, 0.71, null, 2.56, 4.11, 1.99, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [1.19, 1.4, 1.67, 1.65, null, 0.7, 0.66, 0.21, null, 0.66, null, 0.21, null, null, 0.45, null, null, null, 0.4, 0.51, null, 0.98, 1.12, 0.67, 0.96, 0.74, 0.75, null, 0.13, 0.16, 0.58, null, 0.24, null, null, null, null, 0.71, 0.91, null, 0.62, 0.67, null, 0.09, 0.87, null, null, 0.52, null, 0.64, 0.62, 0.61, 0.57, null, null, 0.73, 0.46, null, null, 0.2, 0.17, null, 0.29, null, 0.39, 0.25, 1.52, 0.94, null, null, 0.97, null, 1.05, 0.39, 1.13, 1.01, null, 1.62, null, 0.91, null, null, 0.61, 1.04, 0.61, 0.8, null, 0.3, 0.64, 0.47, 0.99, null], [null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, 1.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, null, 1.68, null, null, null, null], [0.93, 2.18, null, 1.02, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, 0.54, null, null, null, null, 0.39, 1.67, null, 0.6, null, null, null, 1.01, null, null, null, null, null, null, null, 1.32, null, null, null, null, null, null, null, null, 1.57, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, 0.86, null, null, null, 2.73, 0.7, null, null, 1.1, null, null, 0.65, 0.97, 0.86, null, 1.17, null, 1.25, null, null, 0.61, 0.78, null, null, null, 0.1, null, null, 0.66, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.53, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.21, 0.78, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.78, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.89, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.77, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.71, null, null], [1.06, 1.06, null, 0.89, null, 0.82, null, 2.55, null, 3.63, null, 0.85, null, null, 1.61, null, null, null, 2.14, 1.02, 3.36, null, 2.49, 1.54, 1.44, 3.35, 0.98, null, 1.74, 0.47, null, null, 3.12, null, null, null, null, 1.31, 1.52, null, 1.14, null, 1.11, 1.93, 1.74, 1.27, null, 1.12, null, 1.6, 1.24, null, 0.86, 0.67, null, 1.77, 1.85, null, null, null, 1.18, 2.71, 1.59, null, 2.15, 1.85, 0.61, 1.25, null, 2.87, 1.93, null, 1.32, 1.82, 1.94, 1.87, null, 1.03, null, 0.79, 0.94, null, 1.73, 0.86, null, 2.0, null, 0.89, 1.17, null, 1.16, 0.66], [2.52, null, null, 0.51, null, null, 0.22, null, null, null, null, null, null, null, null, null, 0.35, null, null, 0.38, null, null, null, null, null, null, 0.15, null, 0.27, null, null, 2.04, null, null, null, null, null, null, null, null, null, 0.34, null, null, 0.43, null, null, null, null, null, 1.24, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, 0.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.63, null, null, null, null, null, null, 1.78, null, 0.59, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [2.65, 1.12, null, 0.64, 3.41, null, 0.77, 3.83, null, 1.65, null, null, 4.81, null, null, null, null, null, 3.35, 1.28, null, null, null, null, 1.2, null, 4.15, null, 2.68, 0.78, 2.32, null, null, null, null, null, null, 2.38, 2.03, null, 1.04, null, 0.5, 0.88, null, 2.88, null, 0.37, null, 1.6, 2.33, 3.79, 1.44, 0.48, null, 1.66, 1.16, 2.71, 6.63, 3.04, 1.69, 1.13, 0.72, null, 0.98, 1.85, null, 3.12, null, 11.46, 0.69, null, null, 1.3, 0.81, 0.72, null, 0.73, 7.12, 1.14, null, 5.36, 1.73, 5.62, 0.51, 0.67, null, null, null, 1.76, null, 1.77], [null, 6.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.07, null, null, null, null, 2.79, null, null, null, null, null, null, 1.59, null, null, null, null, null, null, null, 2.13, null, null, null, null, null, null, null, null, 2.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.87, null, null, null, null, null, null, null, null, null, null, 1.04, null, null, null, null, null, 2.61, null, null, 1.56, null, null, null, null, null, null, null, 2.48, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, 18.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.04, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 21.13, null, null, 19.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 19.5, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.6, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.4, 0.17, null, null, null, null, 0.99, 0.43, null, null, null, null, null, null, null, null, null, null, 0.27, 0.51, null, null, null, 0.19, null, 0.74, 0.15, null, 0.27, null, 0.29, null, null, null, null, null, null, null, 0.2, null, 0.62, 0.5, 0.91, 0.61, 1.08, null, null, 0.3, null, null, 0.31, null, null, 0.76, null, 0.31, null, 0.41, null, 0.2, 0.17, null, 0.58, null, null, 1.6, 0.61, 0.47, null, null, 0.14, null, 0.53, null, null, null, null, 0.29, null, null, null, null, null, 0.17, null, null, 0.16, null, null, 0.71, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.66, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.29, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.56, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 0.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.75, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null], [0.66, 1.57, null, 0.89, null, 0.23, null, null, null, null, null, null, null, null, 0.45, null, null, null, 0.27, null, null, null, 0.5, 0.39, 0.72, null, 0.75, null, null, null, 0.29, null, null, null, null, null, null, null, 0.81, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, 0.31, 0.46, null, null, 0.2, null, null, 0.29, null, null, null, 2.12, 1.01, null, null, 0.69, null, 0.53, 0.26, 0.81, 0.72, null, 1.03, null, 0.23, null, null, 0.52, 1.12, 0.2, null, null, 0.1, null, null, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, 4.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.53, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.93, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.69, 0.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.66, null, null, null, null, null, 1.3, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null], [0.27, null, null, 0.76, null, 0.7, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, 0.72, null, null, null, null, null, null, null, null, null, null, 0.67, null, 0.18, 0.43, null, null, null, null, null, 0.31, null, 1.15, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, 0.23, null, null, null, null, null, null, 0.31, null, 0.64, 0.24, 0.33, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, 0.1, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, 0.62, null, null, null, null, null, null, null, 2.14, null, null, 0.94, null, 0.5, null, null, null, null, null, null, null, null, null, null, 0.38, null, 0.1, null, null, null, null, null, 0.79, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, 1.67, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.95, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, 0.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.75, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.35, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, 0.21, null, null, 0.44], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, 0.23, null, null, 0.1, null, null, null, null, null, null, 0.3, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, 0.09, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 100.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null], [null, 0.95, 0.83, null, null, 0.94, null, 2.34, null, 1.98, null, 0.64, null, null, 0.54, null, null, null, 1.87, null, null, 0.24, 4.36, 4.05, 0.96, 1.86, 0.15, null, 0.8, 0.47, 0.43, null, 2.4, null, null, null, null, 0.71, 1.12, null, 0.62, null, 0.61, 2.02, null, 0.69, null, 3.59, null, 2.56, null, null, 1.15, 0.57, null, 1.77, 0.46, 0.95, 0.38, null, 1.85, 0.68, 0.58, null, 1.17, 1.98, null, 0.16, null, 0.57, 2.34, null, 0.53, 2.21, 2.1, 2.31, 0.43, null, 1.39, 2.72, 0.63, null, 1.64, 0.35, 1.02, 1.73, 1.26, 0.59, null, null, 1.99, 0.44], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.94, 7.19, 17.05, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 8.51, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.5, null, null, null, null, null, 1.7, null, 2.64, null, null, null, null, null, null, null, null, 1.47, null, null, null, null, 0.96, 1.44, null, null, null, null, null, null, 2.72, 1.92, null, null, null, null, null, 1.02, null, null, null, null, 0.7, null, null, null, 0.82, null, null, null, null, null, null, null, 1.04, null, null, null, null, 0.67, null, null, null, 1.57, 0.99, null, null, null, null, 1.1, null, null, 1.04, 1.94, 1.44, null, null, null, null, null, null, 0.87, null, null, 1.33, null, null, null, null, null, null], [null, null, null, null, null, null, null, 1.28, null, 1.98, 3.19, null, 1.92, null, null, null, null, null, null, 0.77, null, 1.22, 0.5, null, null, null, null, null, 1.21, 0.94, null, null, null, null, null, null, null, 0.71, null, null, 0.62, null, 0.61, null, null, 0.69, null, null, null, null, null, null, null, 0.57, null, null, null, 0.27, null, null, null, 0.68, null, null, null, null, null, null, null, 1.72, null, null, null, null, null, null, null, null, 0.46, null, 0.31, null, null, null, null, null, null, null, null, 0.71, null, null], [null, null, null, null, null, null, null, 1.28, null, 1.65, 2.66, null, 2.4, null, null, null, null, null, null, 0.77, null, 1.96, 0.87, null, null, null, null, null, 0.67, 1.09, null, null, null, null, null, null, null, 1.31, null, null, 1.04, null, 0.81, null, null, 0.69, null, null, null, null, null, null, null, 1.05, null, null, null, 0.68, null, null, null, 0.56, null, null, null, null, null, null, null, 1.72, null, null, null, null, null, null, null, null, 1.02, null, 0.94, null, null, null, null, null, null, null, null, 0.82, null, null], [null, null, null, null, null, null, 0.99, null, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, null, null, null, null, null, null, null, null, 4.08, null, null, null, null, null, null, null, null, null, null, null, 0.79, null, null, null, 0.07, null, 0.96, null, 1.82, null, null, null, null, null, null, 1.14, null, null, null, null, null, 1.17, 0.74, null, null, null, null, null, null, null, 0.39, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, 0.96, null, 0.99, null], [null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, 0.9, null, null, null, null, null, null, null, null, null, null, 0.74, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.5, null, null, null, null], [0.27, null, null, 1.14, null, 1.06, null, null, null, null, null, null, 0.96, null, null, null, null, null, 1.74, 1.15, null, 0.98, null, null, null, null, null, null, null, null, 0.58, null, 0.48, null, null, null, null, null, null, null, null, 1.51, null, null, null, null, null, null, 2.67, null, 1.4, null, 3.16, null, null, null, null, null, 1.7, 0.41, null, null, null, null, 1.37, null, null, null, 4.62, 2.58, 1.24, null, null, null, 1.13, 1.01, null, null, null, 1.02, null, null, null, null, null, null, 0.71, null, null, 1.06, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, 1.06, 0.21, null, null, 0.18, null, null, null, 0.4, null, 4.2, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.09, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.46, null, null, null, null, null, null, null, 0.39, null, null, null, 0.84, null, null, null, null, 0.78, null, null, null, null, null, 0.23, null, null, null, null, null, null, 0.31, null, null, 0.24, null, null], [null, 0.95, null, null, null, null, 0.11, 2.13, null, 1.98, 0.53, 0.11, null, null, 0.09, null, null, null, 2.41, null, null, 1.96, 1.5, 1.35, 2.15, 0.19, 1.06, 0.75, 0.27, 1.4, 0.14, 2.04, 2.4, null, null, null, null, 0.12, 2.03, null, 0.1, 0.17, 0.1, 2.63, null, 0.12, 1.19, 4.79, null, 0.16, 0.16, 2.28, 0.29, 0.1, null, 1.87, 0.46, 0.14, null, null, 1.35, 0.11, 0.43, 0.52, 16.24, 1.48, null, 0.08, 0.42, null, 2.21, null, 0.13, 1.82, 2.1, 2.31, 3.45, null, 0.09, 0.11, 6.73, 2.15, 2.34, 0.09, 0.41, 2.13, null, 0.1, 0.11, null, null, 2.43], [null, null, 3.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.91, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.4, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.57, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 22.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.91, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.36, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.17, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, 2.81, 5.04, null, null, null, null, null, null, null, null, null, null, null, 5.04, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 46.67, 2.18, null, null, 3.98, 2.33, null, null, null, null, 0.78, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.57, null, null, 2.59, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.51, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, null, null, null, 0.13, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, 0.48, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, 0.61, 0.31, null, null, 0.28, null, null, null, 0.32, 0.29, null, 0.29, null, null, null, null, null, 0.35, null, null, null, 0.2, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, 29.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [2.39, null, 10.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 16.67, null, null, null, null, null, null, null, null, 1.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.45, 1.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.38, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.98, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.51, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.25, null, null, null, null, null, null, null, null, 0.47, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, null, null, null, null, null, null], [2.25, 0.11, 1.67, 1.14, null, 0.82, 0.88, null, null, null, null, 4.48, null, null, 1.79, null, null, null, null, 2.17, null, 1.22, 3.49, null, 1.2, 7.26, 3.92, null, null, 2.03, null, null, null, null, null, null, null, null, null, null, null, 2.86, null, null, 3.9, null, null, null, null, 7.84, 2.02, 0.76, 2.3, null, null, 0.1, 1.16, 1.49, 2.08, 2.03, null, null, null, null, null, 1.73, 4.55, 4.37, null, null, 0.14, null, 1.58, null, 0.16, 0.14, null, 2.64, 2.5, 1.93, 1.88, null, null, 5.45, 0.51, null, 2.2, 1.68, 2.77, 0.59, 2.65, 1.11], [0.8, 0.11, null, null, null, null, null, null, null, null, null, 1.17, null, null, 0.36, null, null, null, null, 0.77, null, 0.49, 1.0, null, null, 1.12, 0.75, null, null, 1.09, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, 0.65, null, null, null, null, 0.64, 0.47, null, 0.86, null, null, null, 0.46, 0.54, 0.76, 0.41, null, null, null, null, null, 0.49, 1.21, 0.7, null, null, null, null, 0.92, null, null, null, null, 0.59, 1.11, 0.45, 0.47, null, null, 0.69, null, null, 0.55, 0.3, 0.64, null, 0.66, 0.44], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, 1.41, null, 2.55, null, null, null, 3.63, null, null, 0.36, null, null, null, null, 5.87, null, null, null, null, null, null, null, null, null, 2.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.62, null, null, null, null, null, null, null, null, 0.63, 2.57, null, 5.17, null, null], [null, null, null, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.15, null, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.59, null, null, null, null, null, null, null, null, null, null, null, 0.94, null, null], [0.27, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, 0.08, null, null, 0.14, null, null, null, null, 0.14, 0.86, null, 0.28, null, null, 0.24, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 10.53, null, null, null, null, 2.93, null, null, null, null, null, null, null, 1.72, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.63, 1.7, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.16, null, null, null, null, null, null, 1.82, 2.03, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, 0.09, null, null, null, null, null, null, 3.18, null, 0.39, null, null, null, null, null, 1.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, 0.86, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, 2.16, null, null, null, 1.1, null, null, null, null, null, null, 0.1, null, null, null, 1.55], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, 0.91, null, null, 0.29, null, null, 2.59, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, 0.27, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.32, null, null, null, null, null, 1.98, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, 0.32, null, null, null], [0.53, null, null, null, null, null, null, null, null, null, null, 1.17, null, null, 0.36, null, null, null, null, 0.51, null, 0.49, 0.25, null, null, 0.74, 0.3, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, 0.87, null, null, null, null, 0.64, 0.31, null, 0.57, null, null, null, 0.46, 0.54, 0.76, 0.41, null, null, null, null, null, 0.49, 1.21, 0.31, null, null, null, null, 0.53, null, null, null, null, 0.59, 0.83, 0.45, 0.63, null, null, 0.35, null, null, 0.47, 0.2, 0.64, null, 0.66, 0.44], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [6.63, 4.03, 14.17, 5.84, 11.38, 5.51, 7.29, 6.38, 8.33, 5.61, 7.98, 2.67, 9.13, null, 5.91, null, 8.42, 15.48, 5.35, 5.11, 12.61, 8.56, 3.74, 3.18, 6.22, 10.24, 5.89, 13.53, 2.01, 7.8, 6.51, 0.68, 6.71, null, 6.91, null, 1.56, 5.0, 5.89, 1.11, 3.43, 6.05, 3.33, 8.77, 5.64, 2.65, 25.0, 3.37, null, 6.56, 5.27, 6.37, 4.6, 3.33, null, 6.55, 11.11, 3.93, 5.87, 6.09, 4.05, 4.29, 4.03, 6.81, 4.89, 5.56, 1.82, 6.4, 6.72, 8.88, 7.17, 16.59, 9.35, 4.82, 6.46, 7.06, 11.21, 3.38, 7.77, 4.2, 7.51, 10.49, 5.97, 6.74, 7.11, 4.93, 6.84, 3.96, 7.26, 5.41, 4.47, 16.59], [null, null, 2.92, 0.13, null, 1.99, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, 0.1, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.65, null, 0.16, null, 0.09, null, null, null, null, null, null, null, null, null], [1.46, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, 0.1, 0.69, 0.41, 0.95, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, 0.93, null, null, null, null, null, null, null, 0.24, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [3.32, null, null, null, null, 1.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.37, null, null, null, null, null, null, null, null, 2.88, null, null, null, null, null, 1.09, null, null, null, null, null, null, null, null, null, 1.09, null, null, 1.52, null, null, null, null, 1.44, 9.05, null, 2.22, null, null, 2.38, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.36, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.36, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, 0.09, null, null, null, null, null, null, 0.24, 0.12, null, null, 0.37, null, null, 0.27, 0.31, null, null, null, null, null, null, null, 0.24, null, null, 0.21, null, 0.2, null, null, 0.23, null, null, null, 0.48, null, null, null, 0.19, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, 0.43, null, null, null, 0.31, null, null, null, null, null, 0.24, 0.1, null, null, null, 0.44], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, 0.24, null, 2.87, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, 0.3, 3.04, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, null, 3.63, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.49, null, null, null, null, 0.3, null, null, 0.16, 0.29, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, 0.43, null, null, 0.23, 0.16, null, 0.35, 0.35, 0.2, null, null, null, null, null, null, 0.22], [null, null, 0.42, null, null, null, 0.11, null, null, null, null, null, null, null, 0.09, null, null, null, null, 0.13, null, null, 0.12, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, 0.1, 0.09, null, null, null, 0.15, null, null, null, null, null, null, null, null, 0.23, null, 0.19, null, 0.17, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, 1.72, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null], [0.13, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, 0.49, null, null, 0.24, null, 0.15, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, 0.86, null, null, null, 0.63, null, null, 0.09, null, null, null, 0.3, null, null, null, 0.88], [null, 1.29, null, 1.27, null, 0.59, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.93, null, null, 1.21, null, null, null, null, null, null, null, null, null, null, 1.19, 1.83, null, 1.04, null, 1.01, null, null, 1.15, null, 2.47, null, null, null, null, 7.18, 0.48, null, 1.87, null, null, null, null, null, 1.13, null, null, null, null, null, 3.04, null, null, null, null, null, null, 0.81, 1.73, null, null, null, null, null, null, 2.85, 3.28, null, 2.66, null, null, 3.74, null, 0.83, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.95, null, null, 0.83, null, 0.81, null, null, 0.92, null, null, null, null, null, null, null, 0.38, null, null, null, null, null, null, null, 0.9, null, null, null, null, null, null, null, null, null, null, null, null, 0.65, 0.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.34, null, null, null, 0.59, null, null, null, null, null, 0.53, 5.29, null, null, null, null, 4.76, null, 0.64, null, 1.22, null, null, 1.2, 0.74, null, null, 0.67, 0.47, null, null, null, null, 1.45, null, 6.25, 0.48, null, 0.37, 0.52, null, 2.12, 0.53, null, 0.92, null, null, null, null, null, null, 0.86, 0.57, null, null, 1.62, 1.09, null, 0.3, 0.34, 0.68, 2.88, null, null, 1.73, 0.61, null, null, 2.01, null, null, 2.11, 0.52, null, null, null, null, 0.93, 0.57, 0.63, null, null, null, 4.78, null, 1.1, 1.68, null, 0.59, 0.66, null], [null, 0.22, null, null, null, 0.47, null, null, null, null, null, 0.43, 3.85, null, null, null, null, 4.76, null, 0.51, null, 0.98, null, null, 0.48, 0.74, null, null, 0.54, null, null, null, null, null, 1.45, null, 3.12, 0.83, null, null, 0.62, 0.34, 1.21, 0.7, null, 0.92, null, null, null, null, 0.31, null, 0.57, 0.57, null, null, 1.39, 0.81, null, 0.3, 0.34, 0.68, 1.01, null, null, 0.49, null, null, 2.52, 1.15, null, null, 0.79, 1.04, null, null, null, null, 1.39, 0.68, 0.63, null, null, null, 0.81, null, 0.79, 0.4, null, 0.47, 0.66, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.71, null, null, null, null, null, null, null, null, 0.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, 0.93, null, null, null, null, null, null], [0.27, null, null, 0.25, null, 0.23, 0.33, null, null, null, null, 1.17, null, null, 4.83, null, null, null, null, 0.26, null, null, null, null, null, 1.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, 0.65, null, null, null, 0.67, null, 0.47, 0.3, 0.29, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, 0.23, null, null, null, null, 0.2, null, 0.16, 2.48, 0.32, 0.24, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.6, null, null, null, 0.1, null, null, 3.66, null, null, 4.14, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, 0.87, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, 5.32, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.74, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.39, 1.09, 5.49, null, null, null, null, null, null, null, null, null, null, 6.3, null, null, null, null, null, null, null, null, 1.76, null, null, null, null, null, null, null, 2.52, null, null, null, null, null], [null, 0.84, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.04, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, 2.18, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 2.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.88, null, null, null, null, null, null, null, 1.89, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, 2.18, null, null, null, null, null, null, null, null, null, null, null, 3.51, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.55, 0.09, null, null, null, null, null, null, null, null], [null, 2.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.82, null, null, null, null, null, null, null, 2.79, null, 0.54, null, null, null, null, null, null, null, null, null, 2.34, null, null, null, null, null, null, null, null, 1.35, null, null, null, null, null, null, null, 6.76, null, null, null, null, null, null, null, null, null, null, null, 2.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.58, 1.47, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, 0.28, 2.08, 1.4, 0.98, 0.23, 1.22, 4.68, 33.33, 1.32, 2.13, null, 4.81, 9.76, 0.9, null, 2.11, 5.95, 1.74, 1.02, 2.52, 2.2, 1.75, 1.54, 3.59, 1.49, 0.83, 3.01, 4.02, 3.74, null, 2.72, 0.96, 53.33, 1.45, null, 21.88, 14.88, 1.22, 1.48, 13.41, 2.02, 14.33, 1.14, null, 17.74, 7.14, 0.75, 8.0, 1.12, 1.4, 1.82, null, 13.71, null, 0.21, 1.85, 1.63, 1.89, 7.4, 0.84, 15.14, 2.88, 9.42, 3.52, 1.36, 2.42, 2.11, 4.2, 4.87, 0.69, 3.79, null, 0.78, 1.78, 1.44, 8.19, 0.59, 2.5, 1.48, 2.03, null, 0.87, 2.42, 0.71, 1.33, 1.73, null, 1.07, 0.82, 0.99, 2.21], [1.19, 1.12, 2.08, 0.76, 6.02, 0.94, 1.88, 3.19, null, 2.97, 2.66, 1.6, 5.29, null, 1.88, null, 1.05, 5.95, 2.01, 1.15, 5.04, 2.93, 1.5, 3.85, 2.63, 4.47, 2.57, 5.26, 0.94, 2.03, 3.04, 4.08, 4.08, null, 5.45, null, 1.56, 1.31, 2.24, 0.37, 1.25, 1.51, 0.91, 2.98, 0.87, 1.27, 2.38, 1.57, null, 2.56, 2.02, 2.43, 0.86, 1.05, null, 2.6, 5.32, 2.17, 1.52, 2.64, 2.87, 0.79, 4.76, 4.19, 1.57, 3.46, 0.3, 2.5, 5.88, 6.3, 3.17, 7.58, 2.37, 2.6, 2.42, 3.31, 5.17, 1.03, 3.89, 1.25, 1.41, 3.22, 1.64, 3.2, 2.64, 2.26, 2.04, 3.56, 1.71, 1.41, 0.5, 6.42], [0.4, 0.56, 0.83, 0.76, 4.55, 0.7, 1.22, 1.06, null, 0.66, 1.6, 0.96, 1.92, null, 1.52, null, 1.05, 4.76, 0.8, 1.02, 2.1, 1.71, 1.0, 0.77, 1.2, 2.98, 1.21, 3.76, 0.13, 1.4, 0.43, 0.68, 1.68, null, 2.18, null, 1.56, 0.71, 1.12, null, 0.42, 1.18, 0.2, 0.96, 0.65, 0.35, 2.38, 0.82, null, 2.08, 1.09, 1.21, 0.86, 0.19, null, 0.83, 1.62, 0.68, 0.57, 1.42, 1.18, 0.23, 0.43, 2.62, 0.78, 1.23, 0.3, 1.09, 2.94, 2.87, 0.97, 2.84, 1.32, 0.78, 0.81, 0.86, 3.88, 0.59, 2.22, 0.68, 1.25, 2.03, 0.87, 1.64, 1.12, 1.07, 0.86, 1.19, 1.17, 0.82, 0.5, 3.98], [0.13, null, null, 0.13, 2.93, 0.23, 0.44, null, null, null, 1.06, null, 0.96, null, null, null, null, 2.38, null, null, 0.84, null, null, null, 0.24, null, null, 1.5, null, 0.47, null, null, null, null, 0.73, null, null, null, null, null, null, 0.34, null, 0.18, 0.43, null, null, null, null, null, 0.47, 0.15, null, null, null, 0.1, 0.46, null, null, 0.1, null, null, null, 1.05, null, 0.25, null, null, null, null, null, 0.95, null, null, null, null, null, 0.15, null, 0.23, null, null, null, null, 0.81, null, null, 0.1, 0.32, 0.12, null, null], [0.13, 0.28, 2.5, 0.38, 2.28, 0.35, 0.88, 0.43, 8.33, 0.99, 1.6, 0.32, 0.96, null, 0.63, null, 1.05, null, 1.07, 0.64, 2.1, 0.98, 0.25, 0.39, 0.48, 1.49, 0.6, 12.78, 1.61, 0.62, 1.16, 0.68, 0.72, null, 1.09, null, 1.56, 0.24, 0.81, 0.37, 0.73, 0.67, 0.71, 0.96, 1.08, 0.92, 1.19, 0.52, null, 1.12, 1.09, 0.91, 0.29, 1.33, null, 1.77, 1.85, 1.09, 1.52, 0.91, 1.35, 0.23, 1.01, 1.05, 0.39, 1.36, null, 0.94, 1.26, 1.15, 0.41, null, 0.66, 1.04, 0.97, 1.01, null, 0.59, 1.2, 0.45, 0.78, 0.83, 0.52, 0.69, 0.71, 0.4, 1.97, 0.1, 0.64, 0.71, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.78, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [8.75, 3.75, 8.75, 5.34, 36.75, 10.2, 17.02, 14.26, null, 16.5, 14.36, 4.8, 22.6, null, 10.03, null, 15.44, 28.57, 10.58, 4.34, 31.93, 7.09, 5.49, 6.84, 8.61, 15.46, 8.38, 37.59, 4.69, 4.84, 14.04, null, 13.67, null, 32.73, null, null, 10.48, 7.93, null, 7.59, 10.25, 9.08, 14.47, 11.28, 4.72, 28.57, 5.54, null, 9.76, 10.39, 9.86, 5.46, 9.24, 13.33, 13.62, 12.27, 8.96, 7.01, 10.65, 9.44, 7.57, 8.93, 19.37, 16.83, 15.56, null, 7.81, 13.03, 14.33, 18.62, 9.0, 8.7, 12.11, 10.18, 12.97, 6.47, 8.37, 9.71, 7.26, 5.16, 20.86, 9.6, 7.52, 15.35, 10.39, 6.92, 8.32, 9.07, 8.7, 2.65, 14.38], [null, 0.11, 1.25, 0.25, null, 0.23, 0.55, 0.43, null, 0.66, 1.06, 0.21, 1.92, 7.32, null, null, null, 2.38, 0.27, 0.26, null, 1.71, 0.87, 0.58, 0.24, 0.37, 0.3, 1.5, 2.14, 0.62, null, 1.36, 0.96, null, null, null, 3.12, 0.24, 0.2, null, 0.94, 0.67, 0.2, 0.53, null, 0.58, 2.38, 0.15, 2.67, null, 0.62, 0.61, null, 1.14, null, 0.21, 1.39, 0.27, 0.76, 0.2, 0.34, 0.23, 1.01, 2.09, 0.98, 0.49, 0.61, 0.31, 2.52, 1.15, 0.55, null, 0.26, 0.39, 0.65, 0.58, 6.9, 0.29, 0.46, 0.45, 0.31, null, 0.17, 0.35, 0.2, 0.27, 0.63, 0.2, 0.32, 0.24, null, 0.44], [1.19, 0.84, 2.92, 2.03, 4.72, 2.23, 2.21, 1.7, null, 1.98, 2.13, 0.85, 2.4, null, 3.49, null, 4.21, 3.57, 1.47, 1.28, 5.46, 2.93, 0.87, 1.16, 1.67, 2.23, 1.28, 4.51, 0.67, 2.18, 1.59, null, 2.16, null, 2.91, null, null, 1.07, 1.32, null, 1.04, 2.52, 1.21, 2.19, 2.39, 0.81, 5.95, 0.97, null, 1.44, 1.55, 1.52, 1.15, 1.24, null, 1.56, 3.7, 0.95, 0.95, 1.32, 1.01, 1.24, 1.01, 2.62, 1.17, 1.6, null, 1.72, 1.68, 0.57, 2.07, 0.95, 3.16, 1.56, 2.1, 2.31, 3.88, 1.17, 1.02, 1.36, 2.97, 1.19, 1.12, 1.64, 2.74, 1.73, 1.89, 1.88, 2.88, 1.53, 0.99, 5.31], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, 0.19, null, null, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, null, null, null, null, 0.95, 0.17, 0.09, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.04, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, 0.22], [0.4, 0.5, null, 1.02, 3.9, 1.29, 2.65, 2.55, 25.0, 4.29, 1.6, null, 11.54, null, 0.54, null, null, null, 5.89, 0.89, 10.08, 1.22, 1.0, 1.54, 0.96, 2.05, 1.13, 14.29, 2.55, 6.24, 1.3, 7.48, 3.6, null, 8.0, null, 3.12, 0.71, 1.42, 2.22, 4.99, 1.68, 1.31, 2.63, 2.6, 7.49, 4.76, 1.72, null, 0.64, 1.4, 2.12, 0.86, 3.62, null, 3.22, 1.39, 4.21, 5.68, 1.83, 3.37, 4.41, 1.59, 4.19, 0.59, 4.32, null, 1.8, 0.84, 8.6, 2.07, 4.74, 0.92, 1.69, 1.45, 5.04, null, 1.62, 0.83, 0.79, null, 9.89, 0.78, 1.73, 2.85, 1.07, 3.77, 1.98, 1.81, 1.53, null, 0.22], [null, null, null, null, null, null, null, null, null, null, null, 2.03, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.88], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, null, null, null, null, null, 0.24, null, null, 0.21, null, 0.2, null, null, 0.23, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 12.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.81, null, null, null, null, 0.86, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.39, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 2.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.58, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.65, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, 4.88, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, 14.63, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, 4.88, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, 0.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, 0.33, null], [null, null, null, null, null, null, null, null, null, 1.32, 2.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.4, null, null, 0.25, null, 0.23, 0.22, null, null, null, null, null, null, null, 0.09, null, null, null, null, 0.13, null, null, null, null, 0.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, 0.22, null, null, null, null, null, 0.16, 0.3, 0.29, null, null, 0.1, null, null, null, 0.1, null, null, null, null, null, null, 0.3, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.29, null, 0.23, null, null, null, 0.09, 0.1, null, 0.08, 0.1, 0.21, 0.12, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.37, null, null, null, null, null, null, null, null, 3.59, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 6.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.67, null, null, 6.92, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.63, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 14.85, 29.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.5, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.22, null, null, null, 0.35, null, 0.64, null, 0.99, null, null, null, null, null, null, null, null, 0.54, null, null, null, 1.37, 1.25, 0.48, null, null, null, 0.54, null, null, null, 0.72, null, null, null, null, 0.36, 0.3, null, 0.31, null, 0.3, 0.53, null, 0.35, null, 1.12, null, null, null, null, 0.57, 0.29, null, 0.42, null, 0.41, null, null, 0.67, 0.34, 0.29, null, 0.59, 0.62, null, null, null, null, 0.69, null, null, 0.78, 0.65, 0.72, null, null, null, 1.02, null, null, 0.43, null, 0.3, 0.4, 0.39, null, null, null, 0.99, null], [null, null, null, null, null, null, null, null, null, null, null, null, 5.77, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.96, null, null, null, null, null, null, 6.17, null, null, null, null, null, null, null, null, null, null, null, 2.29, null, null, null, null, 2.88, null, null, null, null, null, null, null, 4.95, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.74, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.1, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 1.32, 2.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.57, null, null, null, 3.78, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.93, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.69, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.93, 0.34, 3.75, 0.89, null, 0.82, 0.88, 1.49, null, null, null, 0.96, null, null, 0.81, null, null, null, 0.8, 0.89, null, 2.93, 0.5, 0.96, 1.44, 1.12, 0.6, null, 1.34, 1.56, 1.45, 4.08, 1.44, null, null, null, null, 0.83, 0.91, null, 0.73, 1.18, 0.71, null, 2.39, 0.81, null, 0.67, 4.67, 1.28, 1.09, 1.67, 2.01, 0.67, null, 0.73, 1.39, 0.95, 1.33, 0.71, 1.01, 0.79, 1.3, 3.14, 1.17, 0.86, 2.12, 0.55, 3.78, 2.01, 0.97, null, 1.32, 1.04, 1.29, 0.86, 3.02, 1.03, 1.48, 0.79, 1.1, 0.83, 0.78, 0.61, 0.71, 1.2, 0.55, 0.69, 0.85, 0.82, 1.16, 2.21], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, 0.53, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.87, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 6.11, null, null, null, null, null, null, 8.98, null, null, null, null, null, null, null, null, null, null, null, 5.09, null, null, null, null, 4.84, null, null, null, null, null, null, null, 7.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.73, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.22, 0.42, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, 0.35, null, 0.13, null, 0.84, 0.24, 0.25, 0.1, null, 0.19, 0.15, null, 0.54, null, null, null, null, null, 1.82, null, null, 0.24, 0.2, null, 0.21, null, 0.2, null, null, 0.23, 2.38, 0.22, null, 0.32, null, null, null, 0.19, null, null, 0.46, 0.27, 0.38, null, null, 0.23, null, null, null, null, null, 0.16, null, 0.57, null, null, null, null, null, null, 0.43, null, 0.46, null, null, null, 0.17, 0.17, 0.2, 0.13, 0.24, null, null, null, null, null], [null, 0.45, null, null, null, null, null, 3.4, null, null, null, null, null, null, null, null, null, null, 1.2, null, null, null, null, 1.45, null, null, 0.6, null, null, null, null, null, 2.88, null, null, null, null, null, 1.73, null, null, null, null, null, null, null, null, 1.42, null, null, null, null, null, null, null, 1.87, null, null, null, null, 2.53, null, null, null, null, 1.48, null, 0.62, null, null, 2.62, null, null, null, 2.58, 2.31, null, null, null, null, null, null, 0.69, 0.69, 10.47, 2.53, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, 2.98, null, null, null, 1.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.94, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.35, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, 0.76, null, 0.7, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, 0.67, null, 0.09, 0.43, null, null, null, null, null, 0.31, null, 1.15, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, 0.29, null, 0.23, null, null, null, null, null, null, 0.31, null, 0.64, 0.24, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, 0.19, 0.24, 0.74, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, 0.99, 0.3, 0.08, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.69, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.07, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.25, null, 0.23, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.06, null, null, null, null, null, null, null, null, 2.4, null, null, null, null, null, 1.02, null, null, 1.85, null, 0.88, null, null, null, 0.75, 8.0, 1.76, 1.71, null, 3.16, null, null, null, 3.47, null, null, null, null, null, 1.44, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.74, null, null, 1.02, null, null, 0.87, null, null, 1.33, 0.79, null, null, null, 1.66, 2.43], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, 0.42, null, 0.81, 0.35, null, null, null, 0.45, null, null, 0.62, null, null, 1.14, null, null, null, 0.54, null, null, 0.34, null, 0.58, null, null, 0.49, null, 0.31, null, null, 0.28, null, 0.53, null, null, null, null, 0.59, null, null, null, null, null, 0.17, null, null, 0.16, null, null, null, null, null], [0.53, null, 5.83, 0.51, null, 0.47, 0.66, null, null, null, null, null, null, null, null, null, null, null, null, 0.64, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.01, null, null, 1.3, null, null, null, null, null, 0.62, 0.61, 0.86, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.73, null, 0.45, null, null, null, null, 0.51, null, null, null, 0.53, 0.59, 0.83, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.61, null, null, null, null, null, null, null, null, 0.95, null, null, 0.52, null, 0.81, null, null, 0.92, null, null, null, null, null, null, null, 0.76, null, null, null, null, null, null, null, 0.56, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, 4.81, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.77, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 22.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.87, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.89, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.11], [0.27, 0.17, 0.42, 0.13, null, 0.23, 0.11, null, null, null, null, null, null, null, null, null, null, null, 0.13, 0.26, null, null, 0.25, 0.19, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, 0.95, 0.2, null, null, 0.34, null, null, 0.43, null, null, 0.07, null, null, 0.31, 0.15, 0.29, null, null, null, null, null, null, null, null, 0.56, 0.14, null, null, null, null, null, null, null, null, null, 0.26, 0.13, null, null, null, 0.15, null, 0.23, null, null, 0.09, null, 0.1, null, null, null, 0.11, 0.24, 0.5, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, 0.38, null, 0.35, 0.33, 0.21, null, null, null, null, null, null, null, null, null, null, 0.13, 0.13, null, null, null, 0.1, null, 0.37, 0.08, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, 0.31, 0.34, 0.5, 0.26, 0.43, null, null, 0.22, null, null, 0.47, null, 0.57, 0.57, null, 0.21, null, 0.27, null, 0.3, 0.17, null, 0.29, null, null, 0.12, null, 0.16, null, null, 0.14, null, 0.4, null, null, null, null, 0.44, null, 0.11, null, null, null, 0.09, null, null, 0.16, null, 0.32, 0.12, 0.17, null], [0.13, null, null, 0.38, null, 0.35, 0.44, 0.43, null, null, null, null, null, null, null, null, null, null, 0.27, 0.13, null, null, null, 0.19, null, 0.74, 0.15, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, 0.62, 0.34, 1.01, 0.53, 0.65, null, null, 0.45, null, null, 0.78, null, 0.57, 1.14, null, 0.62, null, 0.54, null, 0.41, 0.34, null, 0.58, null, null, 0.25, null, 0.31, null, null, 0.28, null, 0.79, null, null, null, null, 0.73, null, 0.11, null, null, null, 0.17, null, null, 0.16, null, 0.32, 0.12, 0.17, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [4.11, 0.56, null, 4.45, null, 3.75, 4.31, null, null, null, null, null, null, null, null, null, null, null, null, 3.7, null, null, null, 1.25, null, null, 0.38, null, null, null, null, null, null, null, null, null, null, 2.5, 1.32, null, 1.77, 4.2, 1.72, null, 4.56, 2.07, null, 1.05, null, null, 5.43, 5.16, 9.48, 0.67, null, 0.62, null, null, null, 3.85, null, 2.03, null, null, null, null, null, 1.17, null, null, null, null, 2.77, null, 1.13, 1.15, null, 2.35, null, 3.18, null, null, 1.04, 1.21, 4.37, 1.6, null, null, 9.71, 4.58, 3.64, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.92, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.87, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.55, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.52, null, null, null, null, null, null, null, null], [0.13, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, 0.24, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, 0.3, 0.23, null, null, 0.14, null, null, null, 0.16, 0.14, null, 0.15, null, null, null, null, null, 0.26, null, null, null, 0.1, null, null, null, null], [null, null, null, 1.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, 0.1, null, null, null, null], [0.4, null, null, null, null, null, 0.66, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, 0.5, null, null, 0.65, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.71, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.29, null, null, null, null], [0.4, 0.28, null, 0.25, null, 0.23, 0.33, 0.85, null, 1.32, null, null, null, null, null, null, null, null, 0.54, 0.26, null, null, null, 0.39, 0.72, null, 0.08, null, null, null, null, null, 0.96, null, null, null, null, null, 0.41, null, null, 0.34, null, 0.35, 0.65, null, null, 0.3, null, null, 0.31, 0.3, 0.29, null, null, 0.42, null, null, null, 0.2, 0.34, null, null, null, 0.78, 0.49, 0.3, 0.08, null, null, 0.55, null, 0.26, 0.52, 0.65, 0.58, null, 0.29, null, 0.23, null, null, 0.35, 0.09, 0.2, 0.53, 0.16, 0.1, 0.32, 0.24, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.58, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.74, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null, null], [0.53, 0.06, null, 1.02, null, 0.7, 0.66, null, null, null, null, null, null, null, null, null, 0.7, null, null, 0.51, null, 0.49, 1.62, 0.19, null, null, 0.08, null, null, 0.31, null, null, null, null, null, null, null, 0.24, 0.2, null, 0.21, 0.67, 0.2, null, 1.3, 0.23, null, 0.15, null, null, 0.93, 0.61, 1.44, 0.1, null, 0.21, null, null, null, 0.61, null, 0.23, null, null, null, null, null, 0.16, null, null, null, null, 1.84, null, 0.16, 0.14, null, 0.59, null, 0.57, 0.16, null, 0.17, 0.17, 0.3, 0.27, null, null, 1.28, 0.47, 0.99, null], [null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.32, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [1.72, 0.11, null, 3.05, null, 2.93, 8.4, null, null, null, null, null, null, null, null, null, null, null, null, 7.02, null, null, null, null, 5.02, null, null, null, 0.13, null, null, 21.09, null, null, null, null, null, 2.26, null, null, 1.35, 4.03, 0.91, 2.28, null, 1.61, null, null, null, null, 3.26, 5.01, null, 1.71, null, 3.43, null, null, 3.6, 2.84, 3.88, 1.47, null, 17.8, 4.7, null, null, null, null, null, 3.31, null, null, 4.04, 4.04, 3.6, null, 2.94, null, 2.72, null, null, null, null, 4.27, null, 1.49, null, 5.66, 6.82, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.63, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.06, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.69, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 19.4, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.28, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.83, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, 1.26, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, 0.06, 0.42, 0.13, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, 0.76, null, 0.7, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, 0.43, null, null, null, null, null, 0.31, null, 1.15, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, 0.23, null, null, null, null, null, null, 0.31, null, 0.64, 0.24, 0.33, null], [null, 0.06, null, 0.89, null, 0.82, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.71, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [1.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, 0.29, null, null, 0.12, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.84, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.49, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.79, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.79, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.47, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, 0.44], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.72, null, null, null, null, 1.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 1.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, null, 0.29, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, 1.12, null, null, null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, 0.39, null, null, null, null, null, 0.34, null, null, 0.26, null, null, null, null, null, null, null, 0.5, null], [0.53, null, 1.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.76, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.21, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.35, null, null, null, null, null, null, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 9.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.53, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, 0.81, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.16, null, null, null, null, 1.05, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.29, null, null, null, null], [null, 0.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.61, null, null, null, null, 1.16, null, null, null, null, null, null, null, 8.16, null, null, null, null, null, null, 1.22, null, null, null, null, 1.05, null, null, null, 0.9, null, null, null, null, null, null, null, null, null, null, null, null, 2.02, null, 1.73, null, null, 1.48, null, null, null, null, null, null, null, 1.56, null, null, null, null, null, null, null, null, 1.04, null, null, 1.6, null, null, null, null, null, null], [0.8, 6.61, null, 0.76, null, 0.7, 0.99, 2.34, null, 1.98, null, 1.07, null, null, 0.9, null, null, null, 3.35, 0.89, 2.52, null, 4.36, 4.91, 0.96, null, 0.15, null, null, null, null, null, 2.4, null, null, null, null, null, 2.54, null, null, 1.01, null, 2.02, 1.52, null, null, 4.72, null, null, 1.09, 0.91, 0.86, null, null, 1.35, 0.69, 0.81, 1.14, 0.91, 1.85, null, 1.44, null, 1.17, 2.96, null, 0.16, null, 1.72, 2.34, null, 0.53, 3.12, 1.78, 2.02, null, 0.88, 1.2, 6.58, null, 5.01, 2.25, 0.17, 0.81, 1.46, 1.42, 0.99, 0.96, 0.71, 5.3, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.5, null, null, null, null, null, 1.34, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.44, null, null, null, null, null, null, null, null, null, null, null, 2.94, null, 1.21, 0.31, null, null, null, null, null, 1.69, 0.32, 0.58, null, null, 0.19, null, 0.63, null, null, null, null, null, null, null, null, null, null, null], [null, 0.45, null, 1.27, null, 0.7, null, 1.49, null, 0.99, null, 0.96, null, null, 1.25, null, null, null, 1.2, null, null, 0.73, 1.37, 0.87, 0.96, null, null, null, null, 1.72, null, null, 1.44, null, null, null, null, null, 0.61, null, null, null, null, 0.35, null, null, null, 0.52, null, null, null, null, 0.86, null, null, 0.62, null, null, null, null, 0.51, null, null, null, 0.98, 0.86, null, null, null, null, 1.24, null, null, 0.91, 1.13, 1.44, 1.29, null, null, null, 3.44, 1.67, 0.61, null, null, 0.93, 1.02, 1.39, null, null, null, 1.77], [null, 7.17, null, null, null, null, 1.88, null, null, null, null, 0.53, null, null, null, null, null, null, 3.21, null, null, 1.47, 7.98, 6.36, null, 2.23, null, null, 2.68, null, 3.04, 10.2, 6.0, null, null, null, null, 6.67, 4.07, null, 4.47, 3.03, 4.24, 2.89, null, 5.99, null, 5.76, null, 2.4, null, 4.4, 4.89, 3.43, null, null, null, 4.07, 3.22, null, null, 6.1, 9.37, null, 2.15, 1.85, null, null, null, null, null, null, 3.95, 3.78, null, null, null, null, 4.35, 4.77, 7.04, null, 5.45, null, null, 6.26, 3.07, null, 2.13, null, 7.12, 2.21], [null, 4.93, null, null, null, null, 1.99, null, null, null, null, null, null, null, null, null, null, null, 2.01, null, null, null, 7.11, 2.31, null, 2.79, null, null, 1.34, null, 2.03, 9.52, 8.63, null, null, null, null, 6.55, 4.17, null, 6.13, 5.04, 2.72, 2.72, null, 6.34, null, 4.42, null, 5.6, null, 4.86, 2.3, 4.38, null, null, null, 5.02, 2.84, null, null, 6.89, 10.23, null, 2.15, 1.73, null, null, null, null, null, null, 3.16, 1.69, null, null, null, null, 5.83, 3.29, 10.17, null, 4.84, null, null, 7.99, 3.62, null, 2.88, null, 4.64, 0.66], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, null, null, 0.41, null, null, null, null, null, null, 1.21, 0.31, 0.84, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.35, null, null, null, 0.2, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.18, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.64, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 15.35, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.59, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.01, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.5, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.56, null, null, null, null, 3.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.71, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.44, null, null, null, null, 7.33, null, 5.27, 1.06, null, null, null, null, 1.85, null, null, null, 2.87, null, null, null, null, 2.59, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.22, null, null, null, 1.07, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.01, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, 1.3, 0.7, null, null, null, null, 4.26, null, null, 12.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.09, null, null, null, 0.9, null, null, 1.57, null, null, null, null, null, null, null, null, null, null, null, 2.59, null, 0.93, null, null, 0.95, null, null, null, null, null, 0.59, null, null, null, null], [null, null, null, null, 0.33, 0.23, null, null, null, null, 1.06, null, null, 2.44, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, 0.45, null, null, 0.78, null, null, null, null, null, null, null, null, null, null, null, 0.86, null, null, null, null, 0.24, null, null, null, null, null, 0.4, null, null, null, null], [null, null, null, null, 0.33, 0.23, null, null, null, null, 1.06, null, null, 2.44, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, null, null, null, 0.23, null, null, 0.39, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, 0.83, null, null, 0.24, null, null, null, null, null, 0.2, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, 0.12, null, null, 2.0, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, 0.94, null, null, null, null, null, 0.39, null, null, null, 0.44, null, null, null, null, null, 1.38, null, null, 0.24, null, null, null, null, null], [null, 0.11, null, null, null, null, null, 0.43, null, 0.66, null, null, null, null, null, null, null, null, 0.27, null, null, null, null, 0.19, 0.24, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, 0.2, null, null, null, null, 0.18, null, null, null, 0.15, null, null, null, null, null, null, null, 0.21, null, null, null, null, 0.17, null, null, null, 0.39, 0.25, null, null, null, null, 0.28, null, null, 0.26, 0.32, 0.29, null, null, null, null, null, null, 0.17, null, null, 0.27, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.75, null, null, null, null, null, 2.01, null, null, null, 2.64, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.54, null, null, null, null, null, null, null, null, 1.95, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 1.99, null, null, 4.95, 7.98, null, null, null, null, null, null, null, null, null, null, 0.49, 0.12, null, 0.72, 2.79, null, 1.5, 0.27, 0.16, 0.14, null, null, null, null, null, null, 0.12, null, null, 0.1, 2.52, 0.1, null, null, 0.12, 11.9, null, null, 2.4, 2.33, 2.28, null, 0.1, null, null, 2.31, 1.49, null, null, null, 0.11, null, 5.76, 0.39, null, null, null, 7.14, null, null, null, 1.58, null, null, null, null, null, 0.56, 1.7, 0.47, 1.55, null, null, 1.52, null, null, 1.19, 1.92, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [0.66, 5.77, 0.83, 0.13, null, 0.47, 0.44, 1.49, null, 1.32, null, 0.96, null, null, 1.52, null, null, null, 0.94, null, null, null, 1.25, 4.05, 0.96, null, 3.09, null, 0.8, null, 4.2, null, 1.44, null, null, null, null, 0.48, 1.93, null, 0.42, null, 0.4, 1.58, 1.08, 0.46, null, 3.59, null, null, 0.31, 0.61, 0.57, 0.38, null, 0.94, 0.23, 0.54, 0.38, 0.51, 1.01, 0.45, 2.59, null, 0.78, 1.98, 1.52, 3.04, null, 0.57, 1.38, null, null, 2.73, 1.13, 1.3, null, 0.15, 0.46, 3.63, null, null, 2.08, 5.19, null, 0.67, 0.86, 1.09, 0.64, 0.47, 4.3, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.19, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, 0.95, 0.91, null, 0.83, null, 0.91, null, null, 1.04, null, 0.67, null, null, null, null, null, 0.38, null, 0.83, null, null, null, null, null, 0.9, null, null, null, null, null, 0.62, null, null, null, null, null, null, 0.65, 0.58, null, null, null, null, null, null, 0.78, 0.69, null, 1.33, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 6.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.73, null, null, null, null, 1.61, null, null, null, null, null, null, null, null, null, 1.83, null, null, null, null, null, null, null, null, 1.05, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.59, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.3, null, null, 2.4, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.84, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.52, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, 25.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.06, null, null, null, null, null, 0.21, null, 0.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, 1.12, null, null, null, null, null, null, 0.24, null, null, null, null, null, 0.1, null, null, null, null, 0.88, null, null, null, 0.07, null, null, null, null, null, null, null, 0.31, 4.4, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, 0.14, null, null, null, null, null, null, 0.09, null, null, null, 0.94, null, null, null, null, null], [null, null, null, null, null, null, 1.66, 1.7, null, null, null, null, null, null, null, null, null, null, 1.07, null, null, null, null, 0.77, null, 3.17, 0.75, null, null, null, null, null, 0.24, null, null, null, null, 0.36, 0.81, null, 2.39, null, 3.33, 2.54, 1.74, null, null, 1.87, null, null, 2.17, null, null, 3.52, null, 3.22, null, 2.31, null, 0.81, 1.35, null, 2.02, null, null, 4.07, null, 2.03, null, null, 1.1, null, 1.71, null, null, null, null, 2.06, null, null, null, null, null, 0.78, null, null, 0.63, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, 0.4, null, null, 0.46, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.45, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 20.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 13.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 12.6, null, null, null, null, null, null, null, null, 10.71, null, null, 9.77, null, 10.9, null, null, 10.94, null, null, null, null, null, null, null, 9.52, null, null, null, null, null, null, null, 9.49, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.8, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null], [0.13, null, null, 0.25, null, 0.23, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, 0.43, null, null, null, null, null, 0.16, 0.3, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, 0.23, null, null, null, null, 0.1, null, null, null, 0.11, 0.12, 0.17, null], [null, null, null, 0.51, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.93, 2.13, null, 0.51, null, 1.41, null, 2.77, null, 2.97, null, 0.64, null, null, 0.9, null, null, null, 2.28, 1.66, null, null, null, 2.12, 1.91, null, 1.51, null, 0.8, null, 1.74, null, 2.88, null, null, null, null, 0.71, 2.03, null, 0.62, 2.02, 0.61, 1.84, 3.47, 0.69, null, 2.25, null, null, null, null, 0.86, 0.57, null, 2.18, 1.39, 1.49, 2.84, 1.12, 1.01, 1.13, 0.43, null, 1.76, 2.96, 0.3, 1.56, null, 2.87, 2.07, null, null, 1.82, 2.1, 2.45, null, 0.15, 3.52, 1.25, null, 4.65, 1.73, 2.07, null, 1.73, 1.42, 0.69, null, null, 0.5, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 14.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, 2.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.61, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.24, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.38, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 2.46, null, null, null, 0.47, null, null, null, null, null, 0.75, null, null, 0.18, null, null, null, null, null, null, null, null, 0.39, null, null, null, null, 0.54, null, null, null, null, null, null, null, null, 0.48, null, null, null, 0.67, null, null, null, null, null, 1.5, 6.67, 0.32, null, null, null, null, null, null, 0.93, 1.09, null, null, null, 0.45, null, 2.09, null, null, null, null, null, 2.29, null, null, null, null, null, null, null, null, 1.11, null, null, null, null, null, null, null, 0.63, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.17, 0.83, 0.13, null, 0.12, 0.99, null, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, 0.62, 0.87, null, 0.56, null, null, 0.4, null, 0.43, 2.04, 0.72, null, null, null, null, 0.83, 0.3, null, 0.31, 0.5, 0.5, 0.79, null, 0.58, null, 0.67, null, 0.48, null, 1.67, null, 0.29, null, null, 0.46, null, 1.14, 0.41, null, 0.79, 0.43, null, 0.59, 0.37, null, null, 3.78, null, 0.28, null, 0.4, 1.17, null, null, null, null, 0.56, 0.57, 0.47, null, 0.78, null, 0.71, 0.4, 0.47, null, 0.64, null, 0.99, 0.66], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.28, null, null, null, null, null, 0.85, null, 1.32, null, null, null, null, null, null, null, null, 0.54, null, null, null, 0.5, 0.39, 1.44, null, null, null, null, null, null, null, 0.72, null, null, null, null, null, 0.41, null, null, null, null, 0.35, null, null, null, 0.3, null, null, null, null, null, null, null, 0.42, null, null, null, null, 0.34, null, null, null, 0.78, 0.49, null, null, null, null, 0.55, null, null, 0.52, 0.32, 0.29, null, null, null, null, null, null, 0.35, null, null, 0.53, null, null, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, 0.1, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, 0.38, null, null, 0.11, 0.14, null, null, null, null, null, null, null, null, null, 0.53, null, null, null, null, null, 0.09, null, null, null, 0.09, null, null, 0.27, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.6, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.76, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.1, null, null, null, null, null, 2.12, null, null, null, null, 1.11], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.93, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.66, 3.64, null, 0.64, null, 0.82, 0.66, 0.43, null, 0.66, null, 2.24, null, null, 2.24, null, null, null, 0.27, 0.51, 1.68, null, 2.99, 2.02, 0.72, 0.37, 0.3, null, 2.14, null, null, null, null, null, null, null, null, 1.43, 1.02, null, 0.94, 1.68, 1.11, 0.18, 0.87, 1.27, null, 2.47, 11.33, 0.48, 0.62, 0.61, 0.57, 1.14, null, 0.1, 1.62, 1.22, null, null, null, 1.47, 0.58, 8.38, null, null, 0.3, 0.23, null, 3.44, 0.14, null, null, 0.52, 0.32, 0.29, null, 0.73, 3.79, 2.27, 0.31, null, 0.87, 0.26, 0.41, 0.27, 2.04, 1.88, 0.64, 0.47, 1.99, 0.44], [null, 0.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.94, null, null, null, 0.75, null, null, null, 0.53, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.55, null, null, null, null, null, null, null, null, null, null, null, null, 0.63, null, null, null, null, null, 0.63, null, null, null, null, 0.88], [null, 0.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.47, null, null, null, null, null, null, 0.19, 0.83, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.86, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, 0.1, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, 0.07, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, 0.11, null, null, 0.09, null, null, null, null, null, null, null, 0.17, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.54, 0.16, null, null, null, null, null, null, null, 0.36, null, null, 0.21, null, 0.3, null, null, 0.35, 2.38, null, null, null, null, null, null, 0.19, null, null, null, 0.27, null, null, null, 0.23, null, 1.05, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.05, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 1.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.34, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.59, null, null, null, null, null, null, null, null, null, null, 1.3, null, null, null, null, 0.93, null, 3.6, null, 1.56, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.69, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, 0.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.81, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.9, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.8, 0.67, null, 1.02, null, null, null, 1.28, null, null, null, null, null, null, 0.63, null, null, null, 1.2, null, null, null, null, 1.73, 1.44, null, 1.06, null, null, null, 0.14, null, 0.48, null, null, null, null, null, 1.02, null, null, null, null, null, null, null, null, 1.95, null, null, null, null, null, null, null, 0.83, null, null, null, null, 0.67, null, 0.43, null, null, 0.49, 2.42, 1.09, null, null, 1.93, null, null, null, 1.94, 1.73, null, 1.17, null, 0.91, null, null, 0.69, 1.04, 0.41, 0.8, 0.47, null, null, null, 0.99, null], [null, null, 0.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null], [1.86, 0.45, null, null, null, null, null, null, null, null, null, null, null, null, 0.98, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, 0.6, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.11, null, null, null, null, null, null, 1.29, 1.87, null, null, null, null, 0.16, null, null, 0.17, null, null, 0.39, null, null, null, null, 0.22], [0.13, null, null, 0.13, null, 0.12, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.22, null, null, null, null, null, 0.31, 0.3, 0.29, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, 0.34, null, null, null, null, 0.2, null, null, null, 0.32, 0.35, null, null], [0.53, null, null, 0.51, null, 0.23, 0.66, null, null, null, null, null, null, null, null, null, null, null, null, 0.51, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, 1.3, null, null, null, null, null, 0.62, 0.61, 0.57, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.59, null, 0.23, null, null, null, null, 0.41, null, null, null, 0.64, 0.47, 0.66, null], [null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null], [13.93, null, null, 13.47, null, 14.65, 0.33, null, null, null, null, 3.42, null, null, 4.21, null, null, null, null, 0.26, null, 2.44, null, null, 21.53, null, 10.64, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, 0.43, null, null, null, null, null, 0.31, 0.3, 0.29, null, null, 8.21, null, null, null, 12.07, null, null, null, null, null, null, 41.82, 9.06, 2.52, null, 15.45, null, null, null, 16.64, 10.95, null, 25.4, null, 0.23, null, null, null, 10.54, 0.2, null, null, 4.65, 0.32, 0.24, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null], [null, 0.06, 5.83, 2.92, null, 2.58, null, 0.21, null, 0.33, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, 2.41, 0.24, 0.19, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, 0.09, null, null, null, 0.07, null, 0.16, null, 2.28, null, null, null, 0.21, 1.85, null, null, 4.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, 0.14, null, null, null, null, null, null, 1.56, null, 0.91, 0.13, 0.08, null, 0.64, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, 0.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, 0.1, null, null, null, null], [null, 0.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, 0.1, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, 0.07, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.14, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, 0.11, null, null, 0.09, null, null, null, null, null, null, null, 0.17, null], [0.27, null, null, 0.76, null, 0.7, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, 0.67, null, 0.09, 0.43, null, null, null, null, null, 0.31, null, 1.15, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, 0.23, null, null, null, null, null, null, 0.31, null, 0.64, 0.24, 0.33, null], [0.13, null, null, null, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, 0.22, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null], [3.05, 0.11, null, 3.18, null, 2.81, 3.65, null, null, null, null, null, null, null, null, null, null, null, null, 3.19, null, null, 0.25, null, 0.48, 0.37, 0.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.37, null, null, 5.64, null, null, null, null, 0.32, 4.03, 3.64, 4.31, null, null, null, null, 0.27, null, 0.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.52, null, 2.84, null, null, null, null, 2.24, null, null, null, 3.2, 2.82, 3.97, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 0.66, 1.06, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 0.99, 1.6, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, 0.82, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.73, null, null, null, null, null, null, null, 6.3, null, null, null, null, null, null, null, null, 1.03, null, null, null, null, null, null, null, null, 4.01, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 2.45, null, null, 0.81, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.87, null, null, null, null], [null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, 0.13, 0.13, null, null, null, null, null, null, null, null, 0.13, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, 0.1, null, 0.43, 0.12, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, 0.11, null, null, null, null, null, 1.25, null, 0.29, null, null, null, 0.26, null, null, null, null, 0.28, null, null, null, null, 1.56, 0.1, null, null, null, null, 0.12, null, null], [null, null, null, 2.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, 0.38, 0.16, null, null, null, null, null, null, null, null, null, 0.18, null, null, null, null, null, null, null, null, null, 0.24, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null, null, null, null, null, null, null, null, 0.61, 0.23, null, null, 0.28, null, null, null, 0.32, 0.29, null, 0.44, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.62, null, null, null, null, 0.35, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, null, null, null, null, null, null, null, 1.87, null, null, null, null, null, null, null, null, null, 0.88, null, null, null, null, null, 1.9, null, null, null, null, null, null, null, null], [null, null, null, 0.76, null, null, null, 0.64, null, null, null, null, null, null, 0.27, null, null, null, 0.54, 0.51, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, null, null, null, null, null, null, null, 1.04, null, 0.3, null, 1.3, 0.69, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, 2.58, null, 2.01, null, null, null, 1.3, null, null, null, null, 0.28, null, null, null, null, 2.16, 0.3, null, null, null, null, 0.47, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.22, null, null, null, null, null, null, null, 0.78, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.24, null, null, null, null, null, null, 0.13, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, null, 0.1, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null], [null, 0.45, null, 1.91, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.45, null, null, null, null, null, null, 1.06, null, null, 1.25, null, null, null, null, null, null, null, null, null, null, null, 2.02, null, 1.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.55, null, null, null, null, null, null, null, null, null, null, null, null, 3.6, null, null, 1.38, null, null, null, null, 3.31, null, null, 8.41], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.98, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 5.93, null, null, null, null, null, null, null, null, null, 19.63, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, 2.5, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, 1.98, 3.19, null, null, null, 2.51, null, 1.05, null, null, null, null, null, null, null, null, 0.37, null, null, 1.07, 2.34, null, null, null, null, null, null, 9.38, null, null, 8.52, null, 0.5, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, 0.81, null, null, null, null, null, null, 1.57, 0.37, null, null, 9.66, null, null, null, null, null, null, null, null, null, null, 0.34, 2.03, null, null, null, 0.61, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 8.0, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.21, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.27, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, 0.78, null, null, null, null, 0.46, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, 2.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.93, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.8, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, 0.47, null, null, null, null, null], [null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.74, null, null, null, null, null, 3.59, null, null, null, null, null, 3.62, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.03, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.56, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 4.57, null, 4.1, null, 7.45, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 8.89, null, null, null, 23.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 4.55, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [2.65, 0.11, 0.83, 1.14, null, 1.17, 1.1, null, null, null, null, null, null, null, null, null, null, null, null, 1.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.34, null, null, 1.95, null, null, null, null, null, 1.4, 0.91, 1.44, null, null, null, null, null, null, 0.71, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.17, null, 1.36, null, null, null, null, 0.81, null, null, null, 1.28, 0.94, 1.32, null], [null, null, null, null, null, 0.23, null, null, null, 0.66, 1.06, null, null, null, null, null, null, null, null, null, null, 0.49, null, 0.19, null, null, null, null, null, 0.31, 0.29, null, null, null, null, null, null, null, 0.2, null, null, null, null, 0.18, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, 0.61, 0.16, null, null, null, null, 0.26, null, null, null, null, null, null, null, 0.31, 0.24, 0.17, null, null, 0.27, 0.16, 0.2, null, null, null, 0.44], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 7.02, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.68, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.18, null, null, null, null, null], [null, 5.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.47, null, null, null, null, 4.43, null, null, null, null, null, null, 4.49, null, null, null, null, null, null, null, 1.93, null, null, null, null, null, null, null, null, 3.97, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.74, null, null, null, null, null, null, null, null, null, null, 1.43, null, null, null, null, 1.02, 3.86, null, null, 2.25, null, null, 0.27, null, null, null, null, 3.81, null], [null, 0.11, null, null, null, null, null, 0.43, null, 0.66, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, 0.48, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, null, null, null, 0.18, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, 0.16, 0.14, null, null, null, null, null, null, 0.17, 0.17, null, 0.27, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.88, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.75, null, null, null, null, null], [null, 0.45, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.96, null, 0.45, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.52, null, null, null, null, null, null, null, 0.62, null, null, null, null, null, null, null, null, null, null, null, 0.47, null, null, null, null, null, null, 0.81, 0.58, null, null, null, null, null, null, 0.78, 0.52, null, 0.8, null, null, null, null, null, null], [null, null, 0.83, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.49, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.12, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.55, null, null, null, null, null], [null, null, 0.42, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.33, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, 0.24, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.45, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.86, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 26.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, 0.58, null, null, null, null, null, null, null, null, null, 0.83, null, 1.61, 0.7, null, null, null, 0.9, null, null, 1.24, null, null, 2.29, null, 0.83, null, 1.09, null, null, 0.67, null, 1.15, null, null, 0.99, null, 0.62, null, null, 0.55, null, 0.92, null, null, null, null, 1.17, null, null, null, null, null, 0.35, null, null, 0.31, null, null, null, null, null], [null, null, null, null, null, null, 0.44, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, null, 0.58, null, null, null, null, null, null, null, null, null, 0.83, null, 1.61, 0.7, null, null, null, 0.9, null, null, 1.24, null, null, 1.81, null, 0.21, null, 1.09, null, null, 0.67, null, 1.15, null, null, 0.99, null, 0.62, null, null, 0.55, null, 1.05, null, null, null, null, 1.17, null, null, null, null, null, 0.35, null, null, 0.31, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, 0.54, null, null, null, null, null, null, null, null, null, null, null, 0.42, null, 0.4, null, null, 0.46, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, null, null, 0.45, null, null, null, null, 0.91, null, null, null, 0.41, null, null, null, 0.48, 0.43, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.21, null, 1.21, null, null, null, null, null, null, null, null, null, null, null, 1.14, null, 1.21, null, null, 1.38, null, null, null, null, null, null, null, 1.33, null, null, null, null, null, null, null, 1.13, null, null, 0.78, null, 4.55, null, null, null, 0.97, null, null, null, 2.75, 2.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, 0.64, null, null, 0.54, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.13, null, null, null, null, null, null, null, null, null, 1.6, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.09, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.67, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 27.41, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 46.67, null, null, null, null, null, 18.52, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.08, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, 0.25, null, 0.23, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.65, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.66], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.33, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.19, null, null, null, null, null, 0.94, null, null, null, null, null, null, null, null, 0.41, null, null, null, null, null, null, null, null, 0.3, null, null, null, null, null, null, null, 0.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.35, null, null, null, null, null, null, null, null, null], [null, 0.56, null, null, null, null, null, 1.06, null, 0.66, null, 0.32, null, null, 0.27, null, null, null, 0.8, null, null, 0.98, 1.12, 0.77, 0.96, 2.23, 0.6, null, 0.8, 1.25, null, null, 1.2, null, null, null, null, 0.71, 0.2, null, 0.62, null, 0.61, 0.44, null, 0.69, null, 0.45, null, 1.76, null, null, null, 0.57, null, 0.31, null, null, null, null, 0.17, 0.68, null, null, 0.98, 0.62, 1.82, 0.62, null, null, 0.69, null, 0.79, 0.52, 0.81, 0.72, 1.72, null, null, null, 1.56, null, 0.52, 0.52, null, 0.8, 1.18, 0.3, null, null, null, 1.77], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.4, null, null, null, null], [0.13, 0.06, 0.42, 0.25, null, 0.23, 0.11, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, 0.25, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.3, null, null, 0.34, null, null, 0.43, null, null, null, null, null, 0.31, 0.3, 0.29, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.26, null, null, null, null, 0.29, null, 0.23, null, null, null, null, 0.2, null, 0.08, null, 0.21, 0.12, 0.33, null], [null, 1.01, 0.42, null, null, null, null, 0.64, null, 0.66, null, 0.53, null, null, 0.63, null, 1.4, null, 0.54, null, 0.84, 0.24, 3.74, 0.19, 0.48, null, 0.75, null, 0.27, 0.31, 0.29, null, 0.48, null, null, null, null, 0.24, 0.2, null, 0.21, null, 0.2, 0.18, null, 0.23, null, 0.15, null, 0.16, null, null, 0.29, 0.19, null, 0.31, 0.23, 0.27, null, null, 0.34, 0.23, null, null, 0.39, 0.62, 1.21, 0.86, null, 0.57, 0.55, null, 3.16, 0.52, 0.48, 0.29, 0.43, null, 0.46, null, 0.31, null, 0.17, 0.86, 0.2, 0.27, 1.49, 0.59, 0.32, null, null, 0.44], [1.86, null, null, 0.64, null, 2.46, 2.65, null, null, null, null, null, null, null, null, null, null, null, null, 2.55, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.01, null, null, 0.87, null, null, null, null, null, 3.26, 2.88, 2.01, null, null, null, null, null, null, 2.03, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.73, null, 1.93, null, 6.67, null, null, 2.34, null, 2.36, null, 3.09, 2.12, 2.15, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.34, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.23, null, 5.84, null, null, null, null, 0.16, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.46, 0.27, 0.38, null, null, null, null, null, null, null, null, null, null, 0.57, null, null, null, null, null, null, null, null, 0.46, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.83, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.34, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 19.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.78, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, 1.46, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.44, null, null, 2.15, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 3.7, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 47.87, null, null, null, null, null, null, null, null, null, null, 0.35, null, 0.41, null, null, null, null, null, null, null], [null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, null, null, null, null, null, 2.18, null, null, null, null, 0.37, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.51, null, null, null, null, null, null, null, null, 0.84, null, null, 2.84, null, 0.39, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 2.18, null, null, null, null, null, null, null, null, 0.31, null, null, null, null, null, 3.94, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 1.14, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.25, 0.48, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, null, null, null, 0.2, null, null, 0.23, null, 0.3, null, null, null, null, null, null, null, null, null, null, null, 0.2, null, 0.56, null, null, null, null, null, null, null, null, 0.28, null, null, 0.65, null, null, null, null, null, 0.34, null, null, 0.17, null, 0.81, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.1, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.12, null, 0.08, null, null, null, null, 0.26, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null], [0.27, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.48, null, 0.08, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.61, 0.16, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, 0.17, null, null, null, 0.2, null, null, null, null]]}],                        {"coloraxis": {"colorbar": {"title": {"text": "percent"}}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "height": 8000, "margin": {"t": 60}, "template": {"data": {"bar": [{"error_x": {"color": "#f2f5fa"}, "error_y": {"color": "#f2f5fa"}, "marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "rgb(17,17,17)", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "baxis": {"endlinecolor": "#A2B1C6", "gridcolor": "#506784", "linecolor": "#506784", "minorgridcolor": "#506784", "startlinecolor": "#A2B1C6"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"line": {"color": "#283442"}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"line": {"color": "#283442"}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#506784"}, "line": {"color": "rgb(17,17,17)"}}, "header": {"fill": {"color": "#2a3f5f"}, "line": {"color": "rgb(17,17,17)"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#f2f5fa", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#f2f5fa"}, "geo": {"bgcolor": "rgb(17,17,17)", "lakecolor": "rgb(17,17,17)", "landcolor": "rgb(17,17,17)", "showlakes": true, "showland": true, "subunitcolor": "#506784"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "dark"}, "paper_bgcolor": "rgb(17,17,17)", "plot_bgcolor": "rgb(17,17,17)", "polar": {"angularaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "radialaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "yaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}, "zaxis": {"backgroundcolor": "rgb(17,17,17)", "gridcolor": "#506784", "gridwidth": 2, "linecolor": "#506784", "showbackground": true, "ticks": "", "zerolinecolor": "#C8D4E3"}}, "shapedefaults": {"line": {"color": "#f2f5fa"}}, "sliderdefaults": {"bgcolor": "#C8D4E3", "bordercolor": "rgb(17,17,17)", "borderwidth": 1, "tickwidth": 0}, "ternary": {"aaxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "baxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}, "bgcolor": "rgb(17,17,17)", "caxis": {"gridcolor": "#506784", "linecolor": "#506784", "ticks": ""}}, "title": {"x": 0.05}, "updatemenudefaults": {"bgcolor": "#506784", "borderwidth": 0}, "xaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "#283442", "linecolor": "#506784", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "#283442", "zerolinewidth": 2}}}, "xaxis": {"anchor": "y", "domain": [0.0, 1.0], "title": {"text": "visited page"}}, "yaxis": {"anchor": "x", "autorange": "reversed", "domain": [0.0, 1.0], "title": {"text": "third-party request"}}},                        {"responsive": true}                    ).then(function(){


                        })                };                });            </script>        </div>


#### oh, you made it this far?

Just one more puzzle piece.

> Mit der anonymen Messung bieten wir Ihnen das optimale Fundament, um Ihre Nutzungswerte mit 100% Datenschutzkonformität – ohne Opt-In-Pflicht – zuverlässig zu erheben.

That's what `https://www.infonline.de/` explains. They are behind `ioam.de` which lurks on over 70 websites in this dataset. So here is one of the scripts they deliver. In constrast to most other scripts it is not obfuscated. 'Guess because they are *100% data protecion compliant*. 




    https://script.ioam.de/iam.js
    /*DO NOT HOST THIS SCRIPT ON YOUR OWN SERVER*/
    var szmvars = "";
    (function(global) {
      var iomnames = "iom".split(',') || ['iom'];
      iomnames = iomnames.length > 4 ? iomnames.slice(0, 3) : iomnames;
      for (var i = 0, iLen = iomnames.length; i < iLen; i +=1) {
        global[iomnames[i]] = (function () {
          var dummySite = "dummy",
              baseUrlDE = "de.ioam.de/tx.io",
              baseUrlLSO = "de.ioam.de/aid.io",
              optinUrl = "de.ioam.de/optin.php?re=",
              qdsUrl = "irqs.ioam.de",
              deBaseUrl = ".ioam.de/tx.io",
              deBaseUrlLSO = ".ioam.de/aid.io",
              deOptinUrl = ".ioam.de/optin.php?re=",
              deSubdomain = ["imarex"],
              cntBaseUrl = ".iocnt.net/tx.io",
              cntBaseUrlLSO = ".iocnt.net/aid.io",
              cntOptinUrl = ".iocnt.net/optin.php?re=",
              cntQdsUrl = "irqs.iocnt.net",
              cntSubdomain = ["at"],
              eventList = ["", "inst", "init","open", "clse", "play", "resm", "stop", "fowa", "bakw", "recd", "paus", "forg", "bakg", "dele", "refr", "kill", "view", "alve", "fini", "mute", "aforg", "abakg", "aclse", "sple", "scvl", "serr", "spyr", "smdr", "sfpl", "sfqt", "ssqt", "stqt", "soqt", "sofc", "scfc", "scqt", "splr", "spli", "sprs", "spre", "smrs", "smre", "sors", "sore", "sack", "sapl", "sapa", "snsp"],
              LSOBlacklist = [],
              checkEvents = 1,
              tb = 0,
              sv = 1,
              lastEvent = "",
              emptyCode = "Leercode_nichtzuordnungsfaehig",
              autoEvents = {
                onfocus:"aforg",
                onblur:"abakg",
                onclose:"aclse"
              },
              nt = 2,
              cookiewhitelist = '[]'.match(/[A-Za-z0-9]+/g) || [],
              cookieName = "ioam2018",
              cookieMaxRuns = 0,
              socioToken = "9103153f604dfc245e460102ec6ec60b",
              OptoutCookieName = "ioamout",
              frequency = 60000,
              hbiAdShort = 5000,
              hbiAdMedium = 10000,
              hbiAdLong = 30000,
              hbiShort = 10000,
              hbiMedium = 30000,
              hbiLong = 60000,
              hbiExtraLong = 300000,
              heart,
              maxSendBoxes = 10;
    
          var IAMPageElement = null,
              IAMQSElement = null,
              qdsParameter = {},
              qdsPopupBlockDuration = 86400000,
              result = {},
              mode,
              eventsEnabled = 0,
              surveyCalled = 0,
              inited = 0;
    
          var lsottl = 86400000,
              lsottlmin = 180000,
              ioplusurl = "me.ioam.de";
    
          var fpCookieDomain = getFpcd(location.hostname),
              consentVendors = ('[730, 785]'.match(/[0-9]+/g) || []).map(function(vendor) { return parseInt(vendor, 10) }),
              consentMaxCheckIntervals = parseInt('10', 10) || 10,
              consentCheckIntervalLength = parseInt('60', 10) || 60,
              cmpUiShownHandler = false,
              consentCookieExpire = new Date();
          consentCookieExpire.setDate(28);
          var consentCookieOptions = {
                name: 'iom_consent',
                domain: fpCookieDomain.length > 0 ? fpCookieDomain.slice(7, fpCookieDomain.length - 1) : '',
                expires: consentCookieExpire.toUTCString(),
                path: '/'
              };
          function setConsent(ct) {
            processConsent(ct, { vendors: consentVendors, cookie: consentCookieOptions, resultKey: 'ct' }, result);
          }
          function loadConsentFromCookie(options) {
            var value = '';
            var date;
            var valueMatch = document.cookie.match(new RegExp('(^| )' + options.name + '=([^;]+)'));
            var valueParts;
            if (valueMatch) {
              valueParts = valueMatch[2].split('&');
              value = valueParts[0];
              date = valueParts[1];
            }
            return {
              value: value,
              date: date
            };
          }
          function writeConsentToCookie(consent, options) {
            var now = Date.now();
            var cookie = '';
            Object.keys(options).forEach(function(key, index, keys) {
              var option = options[key];
              if (key === 'name') {
                cookie += option + '=' + consent + '&' + now;
                cookie += index < keys.length ? '; ' : ''
              } else {
                if (option) {
                  cookie += key + '=' + option;
                  cookie += index < keys.length ? '; ' : ''
                }
              }
            })
            document.cookie = cookie;
          }
          function checkForConsent(consentString, vendors, vendor, purpose,offset) {
            var result = false;
            if (typeof consentString === 'string' && consentString.length === 2 + vendors.length * 4) {
              var vendorIndex = vendors.indexOf(vendor);
              if (vendorIndex > -1) {
                var start = 2;
                var end = start + ((vendorIndex + 1) * 4);
                var consentVendorPart = parseInt(consentString.slice(start, end), 16);
                var purposeBit = Math.pow(2, (purpose + offset));
                result = (consentVendorPart & purposeBit) === purposeBit;
              }
            }
            return result;
          }
          function processConsent(consentString, consentOptions, iamResultSet) {
            function extractConsentFromCmp(tcData, vendors) {
              function extractPurposes(vendor, hasLegitimateInterest, hasSpecialFeatureOptins) {
                function filter(data) {
                  return function(value) {
                    return data[value] === true;
                  };
                }
                function mapper(offset) {
                  return function(value) {
                    var exp = (parseInt(value) + offset);
                    return Math.pow(2, exp);
                  };
                }
                function merge(purposes1, purposes2) {
                  return purposes1.concat(purposes2.filter(function(item) {
                    return purposes1.indexOf(item) < 0;
                  }));
                }
                var purposes;
                var legitimateInterests = [];
                purposes = Object
                  .keys(tcData.purpose.consents)
                  .filter(filter(tcData.purpose.consents))
                  .map(mapper(-1));
                if (hasLegitimateInterest) {
                  legitimateInterests = Object
                    .keys(tcData.purpose.legitimateInterests)
                    .filter(filter(tcData.purpose.legitimateInterests))
                    .map(mapper(-1));
                }
                if (legitimateInterests.length > 0) {
                  purposes = merge(purposes, legitimateInterests);
                }
                if (hasSpecialFeatureOptins) {
                  purposes = purposes.concat(Object.keys(tcData.specialFeatureOptins)
                    .filter(filter(tcData.specialFeatureOptins))
                    .map(mapper(9)));
                }
                return purposes;
              }
              function createPurposesBitfield(purposes) {
                var result = 0x0000;
                for (var i = 0, iLen = purposes.length; i < iLen; i += 1) {
                  result |= purposes[i];
                }
                return result;
              }
              function convertToConsentString(consent) {
                function padStart(str, size) {
                  while (str.length < size) {
                    str = '0' + str;
                  }
                  return str;
                }
                var result = '';
                for (var i = 0, iLen = consent.length; i < iLen; i += 1) {
                  var hex = consent[i].toString(16);
                  var hexLen = 4;
                  if (i === 0) {
                    hexLen = 2;
                  }
                  hex = padStart(hex, hexLen);
                  result += hex;
                }
                return result;
              }
              var consent = [0x01];
              for (var i = 0, iLen = vendors.length; i < iLen; i += 1) {
                var vendor = vendors[i];
                if (tcData.vendor.consents[vendor] === true || tcData.vendor.legitimateInterests[vendor] === true) {
                  var purposes = [];
                  var hasLegitimateInterests = tcData.vendor.legitimateInterests[vendor];
                  var hasSpecialFeaturesOptins = Object.keys(tcData.specialFeatureOptins).length > 0;
                  purposes = extractPurposes(vendors[i], hasLegitimateInterests, hasSpecialFeaturesOptins);
                  consent.push(createPurposesBitfield(purposes));
                } else {
                  consent.push(0x0000);
                }
              }
              return convertToConsentString(consent);
            }
            function createDefaultConsentString(vendors, hasApi) {
              var result = '';
              for(var i = 0, iLen = vendors.length; i < iLen; i += 1) {
                result += '0000';
              }
              result = (hasApi ? '01' : '00') + result;
              return result;
            }
            function handleConsentLoaded(currentConsentString, options, resultSet) {
              return function(tcData, success) {
                var noop = function() {};
                if (success && ['tcloaded', 'useractioncomplete'].indexOf(tcData.eventStatus) > -1) {
                  var extractedConsentString = tcData.gdprApplies
                    ? extractConsentFromCmp(tcData, options.vendors)
                    : createDefaultConsentString(options.vendors, true);
                  if (extractedConsentString !== currentConsentString) {
                    if (resultSet && options.resultKey) {
                      resultSet[options.resultKey] = extractedConsentString;
                    }
                    writeConsentToCookie(extractedConsentString, consentOptions.cookie);
                  }
                  __tcfapi('removeEventListener', 2, noop, tcData.listenerId);
                } else {
                  var failedConsentString = createDefaultConsentString(options.vendors, true);
                  if (resultSet && options.resultKey) {
                    resultSet[options.resultKey] = failedConsentString;
                  }
                  writeConsentToCookie(failedConsentString, consentOptions.cookie);
                }
              };
            }
            function handleCmpUiShown(currentConsentString, options, resultSet) {
              return function(tcData, success) {
                if (success && tcData.eventStatus === 'cmpuishown') {
                  __tcfapi('addEventListener', 2, handleConsentLoaded(currentConsentString, options, resultSet));
                }
              }
            }
            function hasTcfApi() {
              return '__tcfapi' in window;
            }
            var interval = 0;
            var intervalCount = 0;
            var storedConsentString = loadConsentFromCookie(consentOptions.cookie).value;
            var defaultConsentString = createDefaultConsentString(consentOptions.vendors, hasTcfApi());
            if (hasTcfApi()) {
              if (iamResultSet && consentOptions.resultKey) {
                iamResultSet[consentOptions.resultKey] = storedConsentString || defaultConsentString;
              }
              __tcfapi('addEventListener', 2, handleConsentLoaded((storedConsentString || defaultConsentString), consentOptions, iamResultSet));
              if (cmpUiShownHandler === false) {
                __tcfapi('addEventListener', 2, handleCmpUiShown((storedConsentString || defaultConsentString), consentOptions, iamResultSet));
                cmpUiShownHandler = true;
              }
            } else if (!hasTcfApi()){
              interval = setInterval(function() {
                intervalCount += 1;
                if (hasTcfApi() || intervalCount >= consentMaxCheckIntervals) {
                  clearInterval(interval);
                  processConsent(consentString, consentOptions, iamResultSet);
                }
              }, consentCheckIntervalLength);
            }
            if (consentString && consentString !== storedConsentString && hasTcfApi() === false) {
              writeConsentToCookie(consentString, consentOptions.cookie);
              if (iamResultSet && consentOptions.resultKey) {
                iamResultSet[consentOptions.resultKey] = consentString;
              }
            } else if (!consentString && storedConsentString && hasTcfApi() === false) {
              if (iamResultSet && consentOptions.resultKey) {
                iamResultSet[consentOptions.resultKey] = storedConsentString;
              }
            } else if (!consentString && !storedConsentString && hasTcfApi() === false) {
              writeConsentToCookie(defaultConsentString, consentOptions.cookie);
              if (iamResultSet && consentOptions.resultKey) {
                iamResultSet[consentOptions.resultKey] = defaultConsentString;
              }
            }
          }
          function enableEvents() {
            if ((tb == 1 || result.tb == "on") && result.tb != "off" && !eventsEnabled) {
              eventsEnabled = 1;
              mode = 1;
              for(var e in autoEvents) {
                (function(e) {
                  var oldEvent = window[e];
                  window[e] = function() {
                    if (lastEvent != autoEvents[e]) {
                      lastEvent = autoEvents[e];
                      event(autoEvents[e]);
                    }
                    if (typeof oldEvent == "function") oldEvent();
                  };
                })(e);
              }
            }
          }
    
          function isDoNotTrack() {
            if ((nt & 2) ? ((typeof result.nt == "undefined") ? (nt & 1) : result.nt) : nt & 1) {
              if (window.navigator.msDoNotTrack && window.navigator.msDoNotTrack == "1") return true;
              if (window.navigator.doNotTrack && (window.navigator.doNotTrack == "yes" || window.navigator.doNotTrack == "1")) return true;
            }
            return false;
          }
    
          var getInvitation = function (response) {
            if (response && response.hasOwnProperty("block-status")){
              var isEligibleForInvitation = ( "NONE" === response['block-status'].toUpperCase() );
              if (isEligibleForInvitation) {
                if (IAMQSElement) {
                  IAMQSElement.parentNode.removeChild(IAMQSElement);
                }
                IAMQSElement = createScriptTag(response['invite-url']);
              }
            }
          };
    
          function loadSurvey() {
            szmvars = result.st + "//" + result.pt + "//" + result.cp + "//VIA_SZMNG";
            var sampleType = (result.sv == "i2") ? "in" : result.sv;
            var qdsHost = qdsUrl;
            if (result.cn) {
              sampleType += "_"+result.cn;
              if (result.cn == "at") {
                qdsHost = cntQdsUrl;
              }
            }
    
            qdsParameter = {
              siteIdentifier: result.cp,
              offerIdentifier: result.st,
              sampleType: sampleType,
              pixelType: result.pt,
              contentType: result.cp,
              host: qdsHost,
              port: "",
              isFadeoutFlash: true,
              isFadeoutFrame: true,
              isFadeoutForm: true,
              positionTop: 10,
              positionLeft: 100,
              zIndex: 1100000,
              popupBlockDuration: qdsPopupBlockDuration,
              keysForQueryParam : [
                "offerIdentifier",
                "siteIdentifier",
                "sampleType",
                "pixelType",
                "isFadeoutFlash",
                "isFadeoutFrame",
                "isFadeoutForm",
                "positionTop",
                "positionLeft",
                "zIndex"]
            };
    
            if(typeof window.iam_zindex !== 'undefined') {
              qdsParameter.zIndex = window.iam_zindex;
            }
    
            if(typeof window.iam_fadeout_flash !== 'undefined') {
              qdsParameter.isFadeoutFlash = window.iam_fadeout_flash;
            }
    
            if(typeof window.iam_fadeout_iframe !== 'undefined') {
              qdsParameter.isFadeoutFrame = window.iam_fadeout_iframe;
            }
    
            if(typeof window.iam_fadeout_form !== 'undefined') {
              qdsParameter.isFadeoutForm = window.iam_fadeout_form;
            }
    
            if(typeof window.iam_position_top !== 'undefined') {
              qdsParameter.positionTop = window.iam_position_top;
            }
    
            if(typeof window.iam_position_left !== 'undefined') {
              qdsParameter.positionLeft = window.iam_position_left;
            }
    
            var filterObjectByKeys = function (obj, keysToFilter) {
              var result = {}, key;
              var arrayLength = keysToFilter.length;
              for (var i = 0; i < arrayLength; i++) {
                key = keysToFilter[i];
                if (obj.hasOwnProperty(key)) {
                  result[key] = obj[key];
                }
              }
              return result;
            };
    
            var serializeToQueryString = function (obj) {
              var str = [];
              for (var key in obj)
                if (obj.hasOwnProperty(key)) {
                  str.push(encodeURIComponent(key) + "=" + encodeURIComponent(obj[key]));
                }
              return str.join("&");
            };
    
            var createPopupcheckCookie = function (blockDuration) {
              var blockedUntilDate = new Date();
              blockedUntilDate.setTime(blockedUntilDate.getTime() + blockDuration);
              var expires = "expires=" + blockedUntilDate.toUTCString();
              document.cookie = "POPUPCHECK=" + blockedUntilDate.getTime().toString() + ";" + expires + ";path=/";
            };
    
            var hasPopupcheckCookie = function () {
              var cookie = document.cookie.split(";");
              for (var i = 0; i < cookie.length; i++) {
                if (cookie[i].match("POPUPCHECK=.*")) {
                  var currentDate = new Date();
                  var now = currentDate.getTime();
                  currentDate.setTime(cookie[i].split("=")[1]);
                  var blockedUntilTime = currentDate.getTime();
                  if (now <= blockedUntilTime) {
                    return true;
                  }
                }
              }
              return false;
            };
    
            if (hasPopupcheckCookie()) {
              return;
            }
    
            if (sv && !surveyCalled && result.sv !== "ke" && result.sv === "dz") {
              surveyCalled = 1;
              iam_ng_nxss();
            }
    
            if (sv && !surveyCalled && result.sv !== "ke" && (result.sv === "in" || result.sv === "mo" || result.sv === "i2" )) {
              surveyCalled = 1;
              createPopupcheckCookie(qdsParameter.popupBlockDuration);
              // var protocol = window.location.protocol;
              var protocol = "https:";
              var pathOfCheckInvitation = "identitystatus";
              var queryParameter = filterObjectByKeys(qdsParameter, qdsParameter.keysForQueryParam);
              var queryParameterString = "?" + serializeToQueryString(queryParameter);
              if (window.XDomainRequest && document.documentMode === 9) {
                var checkForInvitationUrl = protocol + '//' + qdsParameter.host + '/' + pathOfCheckInvitation + '/identity.js' + queryParameterString+'&callback=iom.gi&c='+Math.random();
                createScriptTag(checkForInvitationUrl);
              } else {
                var checkForInvitationUrl = protocol + '//' + qdsParameter.host + '/' + pathOfCheckInvitation + queryParameterString+'&c='+Math.random();
                var httpRequest = new XMLHttpRequest();
                httpRequest.onreadystatechange = function () {
                  if (httpRequest.readyState === XMLHttpRequest.DONE && 200 === httpRequest.status) {
                    var response = JSON.parse(httpRequest.responseText);
                    getInvitation(response);
                  }
                };
                httpRequest.open('GET', checkForInvitationUrl, true);
                httpRequest.withCredentials = true;
                httpRequest.send(null);
              }
    
            }
          }
    
          function hash(key) {
            var hash = 0;
            for (var i=0; i<key.length; ++i) {
              hash += key.charCodeAt(i);
              hash += (hash << 10);
              hash ^= (hash >> 6);
            }
            hash += (hash << 3);
            hash ^= (hash >> 11);
            hash += (hash << 15);
            hash = Math.abs(hash & hash);
            return hash.toString(36);
          }
    
          function activeXDetect() {
            var result = "",
                componentVersion,
                components =[
                  "7790769C-0471-11D2-AF11-00C04FA35D02", "89820200-ECBD-11CF-8B85-00AA005B4340",
                  "283807B5-2C60-11D0-A31D-00AA00B92C03", "4F216970-C90C-11D1-B5C7-0000F8051515",
                  "44BBA848-CC51-11CF-AAFA-00AA00B6015C", "9381D8F2-0288-11D0-9501-00AA00B911A5",
                  "4F216970-C90C-11D1-B5C7-0000F8051515", "5A8D6EE0-3E18-11D0-821E-444553540000",
                  "89820200-ECBD-11CF-8B85-00AA005B4383", "08B0E5C0-4FCB-11CF-AAA5-00401C608555",
                  "45EA75A0-A269-11D1-B5BF-0000F8051515", "DE5AED00-A4BF-11D1-9948-00C04F98BBC9",
                  "22D6F312-B0F6-11D0-94AB-0080C74C7E95", "44BBA842-CC51-11CF-AAFA-00AA00B6015B",
                  "3AF36230-A269-11D1-B5BF-0000F8051515", "44BBA840-CC51-11CF-AAFA-00AA00B6015C",
                  "CC2A9BA0-3BDD-11D0-821E-444553540000", "08B0E5C0-4FCB-11CF-AAA5-00401C608500",
                  "D27CDB6E-AE6D-11CF-96B8-444553540000", "2A202491-F00D-11CF-87CC-0020AFEECF20"
                ];
            document.body.addBehavior( "#default#clientCaps" );
            for (var i = 0; i < components.length; i++) {
              componentVersion = document.body.getComponentVersion('{' + components[i] + '}', 'ComponentID');
              if ( componentVersion !== null ) {
                result += componentVersion;
              } else {
                result += "null";
              }
            }
            return result;
          }
    
          function fingerprint() {
            var nav = window.navigator, t = nav.userAgent;
            t += getScreen();
            if (nav.plugins.length > 0 ) {
              for (var i = 0; i < nav.plugins.length; i++ ) {
                t += nav.plugins[i].filename + nav.plugins[i].version + nav.plugins[i].description;
              }
            }
            if (nav.mimeTypes.length > 0 ) {
              for (var i = 0; i < nav.mimeTypes.length; i++ ) {
                t += nav.mimeTypes[i].type;
              }
            }
            if ( /MSIE (\d+\.\d+);/.test(nav.userAgent) ) {
              try {
                t += activeXDetect();
              }
              catch(e) {
                // ignore
              }
            }
            return hash(t);
          }
    
          function createScriptTag(url){
            var el = document.createElement("script");
            el.type = "text/javascript";
            el.src = url;
            var head = document.getElementsByTagName("head")[0];
            if(head) {
              head.appendChild(el);
              return el;
            }
            else return false;
          }
    
          function createScriptTagAsync(url, cb){
            var el = document.createElement("script");
            el.type = "text/javascript";
            el.src = url;
            el.onload = cb;
            el.async = true;
            var head = document.getElementsByTagName("head")[0];
            if(head) {
              head.appendChild(el);
              return el;
            }
            else return false;
          }
    
          function createIamSendBox(url) {
            function appendSendBox(url) {
              var sendBox = document.createElement("iframe");
              sendBox.className = 'iamsendbox';
              sendBox.style.position = 'absolute';
              sendBox.style.left = sendBox.style.top = '-999px';
              sendBox.src = url + '&mo=1';
              document.body.appendChild(sendBox);
            }
            var sendBoxes = document.querySelectorAll('.iamsendbox');
            if (sendBoxes.length < maxSendBoxes) {
              appendSendBox(url);
            } else {
              sendBoxes[0].remove();
              appendSendBox(url);
            }
          }
    
          function transmitData(url, mode) {
            if (url.split("/")[2].slice(url.split("/")[2].length-8) == ".ioam.de" || url.split("/")[2].slice(url.split("/")[2].length-10) == ".iocnt.net") {
              switch (mode) {
                case 1:
                  if (IAMPageElement) {
                    IAMPageElement.parentNode.removeChild(IAMPageElement);
                  }
                  IAMPageElement = createScriptTag(url+'&mo=1');
                  if (!IAMPageElement) (new Image()).src = url+'&mo=0';
                  break;
                case 2:
                  (new Image()).src = url+'&mo=0';
                  break;
                case 3:
                  createIamSendBox(url);
                  break;
                case 0:
                default:
                  document.write('<script src="'+url+'&mo=1"></script>');
              }
            }
          }
    
          function getScreen() {
            return screen.width + "x" + screen.height + "x" + screen.colorDepth;
          }
    
          function arrayContains(arr, obj) {
            var i;
            for (i=0;i<arr.length;i++) {
              if (arr[i]==obj) return true;
            }
            return false;
          }
    
          function transformVar(value) {
            if (!value) value = "";
            value = value.replace(/[?#].*/g, "");
            value = value.replace(/[^a-zA-Z0-9,_\/-]+/g, ".");
            if (value.length > 255) value = value.substr(0,254) + '+';
            return value;
          }
    
          function transformRef(value) {
            if (!value) value = "";
            //value = value.replace(/[?#].*/g, "");
            value = value.replace(/[^a-zA-Z0-9,_\/:-]+/g, ".");
            if (value.length > 255) value = value.substr(0,254) + '+';
            return value;
          }
    
          function getRefHost() {
            var url = document.referrer.split("/");
            return (url.length >= 3) ? url[2] : "";
          }
    
          function buildResult(params) {
            result = {};
            var i;
            for (i in params) {
              if (params.hasOwnProperty(i)) {
                if (i != "cn" || (i == "cn" && (arrayContains(deSubdomain, params[i])) || (arrayContains(cntSubdomain, params[i])))) {
                  result[i] = params[i];
                }
              }
            }
            if (result.hasOwnProperty("fp")) {
              result.fp = (result.fp != "" && typeof result.fp != "undefined") ? result.fp : emptyCode;
              result.fp = transformVar(result.fp);
              result.pt = "FP";
            }
            if (result.hasOwnProperty("np")) {
              result.np = (result.np != "" && typeof result.np != "undefined") ? result.np : emptyCode;
              result.np = transformVar(result.np);
              result.pt = "NP";
            }
            if (result.hasOwnProperty("xp")) {
              result.xp = (result.xp != "" && typeof result.xp != "undefined") ? result.xp : emptyCode;
              result.xp = transformVar(result.xp);
              result.pt = "XP";
            }
            if (result.hasOwnProperty("cp")) {
              result.cp = (result.cp != "" && typeof result.cp != "undefined") ? result.cp : emptyCode;
              result.cp = transformVar(result.cp);
              result.pt = "CP";
            }
            if (result.hasOwnProperty("ms")) {
              result.ms = (result.ms != "" && typeof result.ms != "undefined") ? result.ms : "";
            }
            if (!result.pt) {
              result.cp = emptyCode;
              result.pt = "CP";
              result.er = "N13";
            }
            if (!result.hasOwnProperty("ps")) {
              result.ps = "lin";
              result.er = "N22";
            } else {
              if (!(arrayContains(['ack', 'lin', 'pio', 'out'], result.ps))) {
                result.ps = "lin";
                result.er = "N23";
              }
            }
            result.rf = getRefHost();
            if (!result.hasOwnProperty("sur") || (result.hasOwnProperty("sur") && result.sur != "yes")) {
              result.r2 = transformRef(document.referrer);
            }
            result.ur = document.location.host;
            result.xy = getScreen();
            result.lo = "DE/Berlin";
            result.cb = "001d";
            result.i2 = "001dd5cb280c42a79604dfc24";
            result.ep = parseInt('1645866271', 10);
            result.vr = "423";
            result.id = fingerprint();
            result.st = result.st ? result.st : dummySite;
            if (!result.hasOwnProperty("sc") || (result.hasOwnProperty("sc") && result.sc != "no")) {
              var cookie = getFirstPartyCookie();
              result.i3 = cookie.cookie;
              result.n1 = cookie.length;
            }
            if (((arrayContains(cookiewhitelist, result.st)) || (result.hasOwnProperty("sc") && result.sc == "yes")) && result.i3 == "nocookie") {
              result.i3 = setFirstPartyCookie();
            }
    
            if (!result.hasOwnProperty("cn") && result.st.charAt(2) == "_") {
              var cn = result.st.substr(0,2);
              if (arrayContains(deSubdomain, cn) || arrayContains(cntSubdomain, cn)) {
                result.cn = cn;
              } else {
                result.er = "E12";
              }
            }
    
            // DNT dissemination survey
            try {
              result.dntt = ((window.navigator.msDoNotTrack && window.navigator.msDoNotTrack == "1") || (window.navigator.doNotTrack && (window.navigator.doNotTrack == "yes" || window.navigator.doNotTrack == "1"))) ? "1" : "0";
            } catch(e) {
              // ignore
            }
          }
    
          function event(event) {
            var payLoad = "";
            var i;
            event = event || "";
            stopHeart();
            if (inited && !isDoNotTrack() && (!checkEvents || (checkEvents && arrayContains(eventList, event))) && result.ps !== "out") {
              result.lt = (new Date()).getTime();
              result.ev = event;
              // var proto = ( window.location.protocol.slice(0,4) === 'http' ) ? window.location.protocol : "https:";
              var proto = "https:";
              var baseUrl = baseUrlDE;
              if (result.cn && arrayContains(deSubdomain, result.cn)) {
                baseUrl = result.cn + deBaseUrl;
              } else if (result.cn && arrayContains(cntSubdomain, result.cn)) {
                baseUrl = result.cn + cntBaseUrl;
              }
              if ( !(arrayContains(LSOBlacklist, result.st)) && ( ((/iPhone/.test(window.navigator.userAgent) || /iPad/.test(window.navigator.userAgent)) && /Safari/.test(window.navigator.userAgent) && !(/Chrome/.test(window.navigator.userAgent)) && !(/CriOS/.test(window.navigator.userAgent))) || ( /Maple_201/.test(window.navigator.userAgent) || /SMART-TV/.test(window.navigator.userAgent) || /SmartTV201/.test(window.navigator.userAgent) ) ) ) {
                if (result.cn && arrayContains(deSubdomain, result.cn)) {
                  baseUrl = result.cn + deBaseUrlLSO;
                } else if (result.cn && arrayContains(cntSubdomain, result.cn)) {
                  baseUrl = result.cn + cntBaseUrlLSO;
                } else {
                  baseUrl = baseUrlLSO;
                }
                mode = 3;
                if (result.hasOwnProperty("sur") && result.sur == "yes") {
                  result.u2 = window.location.origin;
                } else {
                  result.u2 = document.URL;
                }
              }
              for (i in result) {
                if (result.hasOwnProperty(i) && i!="cs" && i!="url") {
                  payLoad = payLoad + encodeURIComponent(i).slice(0,8) + "=" + encodeURIComponent(result[i]).slice(0,2048) + "&";
                }
              }
              payLoad = payLoad.slice(0,4096);
              result.cs = hash(payLoad);
              result.url = proto + "//" + baseUrl + "?" + payLoad + "cs=" + result.cs;
              transmitData(result.url, mode);
              if (arrayContains(['play', 'resm', 'alve', 'mute', 'sfqt', 'ssqt', 'stqt', 'sapl', 'snsp'], event) && (mode === 1 || mode === 3) && result.hasOwnProperty('hb')) {
                startHeart();
              }
              return result;
            }
            return {};
          }
    
          function forwardToOldSZM() {
            if (result.oer === "yes" && !window.IVW && !document.IVW) {
              var SZMProtocol = (window.location.protocol.slice(0,4) === 'http') ? window.location.protocol : "https:";
              var SZMComment = (result.co) ? result.co + "_SENT_VIA_MIGRATION_TAG" : "SENT_VIA_MIGRATION_TAG";
              var SZMCode = (result.oc) ? result.oc : ((result.cp) ? ((result.cp == emptyCode) ? "" : result.cp) : "");
              var SZMContType = (result.pt !== null) ? result.pt : "CP";
              (new Image()).src = SZMProtocol + "//" + result.st + ".ivwbox.de/cgi-bin/ivw/" + SZMContType.toUpperCase() + "/" + SZMCode + ";" + SZMComment + "?r=" + escape(document.referrer) + "&d=" + (Math.random()*100000);
            }
          }
    
          function count(params, m) {
            init(params,m);
            return event(result.ev);
          }
    
          function init(params,m) {
            if (!params.cn || params.cn !== 'at') {
              processConsent(params.ct, { vendors: consentVendors, cookie: consentCookieOptions, resultKey: 'ct' }, params);
            }
            // Remove AMP consent string when provided
            if (params.act) {
              delete params.act;
            }
            mode = m;
            buildResult(params);
            if (result.sv) {
              result.sv = (result.sv == "in" && mode == 1) ? "i2" : result.sv;
            }
            if (result.sv && result.sv !== 'ke' && checkForConsent(params.ct, consentVendors, 785, 9, -1) === false) {
              result.sv = 'ke';
            }
            enableEvents();
            loadSurvey();
            checkOptoutCookie();
            inited = 1;
            forwardToOldSZM();
            return {};
          }
    
          function hybrid(params,m) {
            init(params,m);
            var ioam_smi = (typeof localStorage === 'object' && typeof localStorage.getItem === 'function') ? localStorage.getItem("ioam_smi") : null;
            var ioam_site = (typeof localStorage === 'object' && typeof localStorage.getItem === 'function') ? localStorage.getItem("ioam_site") : null;
            var ioam_bo = (typeof localStorage === 'object' && typeof localStorage.getItem === 'function') ? localStorage.getItem("ioam_bo") : null;
            if ( ioam_smi !== null && ioam_site !== null && ioam_bo !== null ) {
              result.mi = ioam_smi;
              result.fs = result.st;
              result.st = ioam_site;
              result.bo = ioam_bo;
              if (result.fs == result.st) {
                result.cp = (result.cp.slice(0,10) !== "___hyb2___") ? "___hyb2___"+result.fs+"___"+result.cp : result.cp;
              } else {
                result.cp = (result.cp.slice(0,9) !== "___hyb___") ? "___hyb___"+result.fs+"___"+result.cp : result.cp;
              }
              return event(result.ev);
            } else if ( ioam_smi !== null && ioam_bo !== null ) {
              return {};
            } else {
              if ( window.location.protocol.slice(0,4) !== 'http' || /IOAM\/\d+\.\d+/.test(window.navigator.userAgent) ) {
                return {};
              } else {
                return event(result.ev);
              }
            }
          }
    
          function setMultiIdentifier(midentifier) {
            if ( localStorage.getItem("ioam_smi") === null || localStorage.getItem("ioam_site") === null || localStorage.getItem("ioam_bo") === null || localStorage.getItem("ioam_smi") !== midentifier ) {
              result.fs = result.st;
              var JsonMIndetifier = null;
              var NewSite = null;
              if ( typeof midentifier === 'string' && typeof JSON === 'object' && typeof JSON.parse === 'function' ) {
                try {
                  JsonMIndetifier = JSON.parse(midentifier);
                  if (JsonMIndetifier.hasOwnProperty( 'library' )) {
                    if (JsonMIndetifier.library.hasOwnProperty( 'offerIdentifier' )) {
                      if ( JsonMIndetifier.library.offerIdentifier ) {
                        NewSite = JsonMIndetifier.library.offerIdentifier;
                      } else {
                        result.er = "JSON(E10): offerIdentifier not valid";
                      }
                    } else {
                      result.er = "JSON(E10): no key offerIdentifier";
                    }
                  } else {
                    result.er = "JSON(E10): no key library";
                  }
                } catch(err) {
                  result.er = "JSON(E10): "+err;
                }
              }
              if ( NewSite !== null ) {
                localStorage.setItem("ioam_site", NewSite);
              }
              result.st = NewSite;
              result.mi = midentifier;
              result.bo = (new Date()).getTime();
              localStorage.setItem("ioam_smi", result.mi);
              localStorage.setItem("ioam_bo", result.bo);
              if (result.fs == result.st) {
                result.cp = (result.cp.slice(0,10) !== "___hyb2___") ? "___hyb2___"+result.fs+"___"+result.cp : result.cp;
              } else {
                result.cp = (result.cp.slice(0,9) !== "___hyb___") ? "___hyb___"+result.fs+"___"+result.cp : result.cp;
              }
              return event(result.ev);
            }
            return {};
          }
    
          if (window.postMessage || window.JSON && {}.toString.call(window.JSON.parse) !== '[object Function]' && {}.toString.call(window.JSON.stringify) !== '[object Function]') {
            var listener = function(msg) {
              try {
                var msgdata = JSON.parse(msg.data);
              } catch(e) {
                msgdata = { type:false };
              }
              if ({}.toString.call(msgdata) === '[object Object]' && msgdata.type == "iam_data") {
                var respObj = {
                  seq : msgdata.seq,
                  iam_data : {
                    st: result.st,
                    cp: result.cp
                  }
                };
                msg.source.postMessage(JSON.stringify(respObj),msg.origin);
              }
            };
            if (window.addEventListener) {
              window.addEventListener("message", listener);
            } else {
              window.attachEvent("onmessage", listener);
            }
          }
    
          function optin() {
            var oiurl = ( window.location.protocol.slice(0,4) === 'http' ) ? window.location.protocol : "https:" + "//" + optinUrl;
            var win = window.open(oiurl, '_blank');
            win.focus();
          }
    
          function startHeart() {
            // IE 9 Compatible
            function heartbeat() {
              return event("alve");
            }
            switch (result.hb) {
              case "adshort":
                frequency = hbiAdShort;
                break;
              case "admedium":
                frequency = hbiAdMedium;
                break;
              case "adlong":
                frequency = hbiAdLong;
                break;
              case "short":
                frequency = hbiShort;
                break;
              case "medium":
                frequency = hbiMedium;
                break;
              case "long":
                frequency = hbiLong;
                break;
              case "extralong":
                frequency = hbiExtraLong;
                break;
              default:
                frequency = 0;
            }
            if (frequency != 0) {
              try {
                heart = setInterval(heartbeat, frequency);
              } catch(e) {
                // pass
              }
            }
          }
    
          function stopHeart() {
            try {
              clearInterval(heart);
            } catch(e) {
              // pass
            }
          }
    
          function stringtohex(str) {
            var res = [];
            for (var n = 0, l = str.length; n < l; n ++) {
              var hex = Number(str.charCodeAt(n)).toString(16);
              res.push(hex);
            }
            return res.join('');
          }
    
          function getUniqueID() {
            var max = 999999999999;
            var min = 100000000000;
            return (Math.floor(Math.random() * (max - min + 1)) + min).toString(16) + (Math.floor(Math.random() * (max - min + 1)) + min).toString(16) + stringtohex(result.cb) + (Math.floor(Math.random() * (max - min + 1)) + min).toString(16);
          }
    
          function expireDays() {
            var max = 365;
            var min = 300;
            return Math.floor(Math.random() * (max - min + 1)) + min;
          }
    
          function getFirstPartyCookie() {
            //FF Patch
            try {
              var cookie = document.cookie.split(";");
              for (var i = 0; i < cookie.length; i++) {
                if (cookie[i].match(cookieName + "=.*")) {
                  var ourcookie = cookie[i].split("=")[1].replace("!", ":");
                  var cookieParts = ourcookie.split(":");
                  var firstCookieParts = cookieParts.slice(0, cookieParts.length - 1).join(':');
                  var lastCookiePart = cookieParts.slice(-1).pop();
                  if (hash(firstCookieParts) === lastCookiePart) {
                    if (!result.hasOwnProperty("i3") || !result.i3) {
                      updateFirstPartyCookie(ourcookie);
                    }
                    return {
                      cookie: ourcookie,
                      length: cookie.length
                    };
                  } else {
                    // checksum failed, cookie not trusted, delete cookie
                    result.er = "N19";
                    try {
                      if (cookieMaxRuns < 3) {
                        cookieMaxRuns++;
                        setFirstPartyCookie(2000);
                      } else {
                        result.er = "N20";
                      }
                    } catch(e) {
                      result.er = "N20";
                    }
                  }
                }
              }
            } catch(e) {
              return {cookie: "nocookie", length: 0};
            }
            return {cookie: "nocookie", length: cookie.length};
          }
    
          function checkFirstPartyCookie() {
            var cookie = getFirstPartyCookie();
            if (cookie.cookie != "nocookie") {
              return true;
            } else {
              return false;
            }
          }
    
          function getFpcd(cd) {
            var ctld ='acadaeafagaialamaoaqarasatauawaxazbabbbdbebfbgbhbibjbmbnbobrbsbtbwbybzcacccdcfcgchcickclcmcncocrcucvcwcxcyczdjdkdmdodzeceeegereseteufifjfkfmfofrgagdgegfggghgiglgmgngpgqgrgsgtgugwgyhkhmhnhrhthuidieiliminioiqirisitjejmjojpkekgkhkikmknkpkrkwkykzlalblclilklrlsltlulvlymamcmdmemgmhmkmlmmmnmompmqmrmsmtmumvmwmxmymznancnenfngninlnonpnrnunzompapepfpgphpkplpmpnprpsptpwpyqarerorsrurwsasbscsdsesgshsiskslsmsnsosrssstsvsxsysztctdtftgthtjtktltmtntotrtttvtwtzuaugukusuyuzvavcvevgvivnvuwfwsyeytzazmzw'.match(/.{1,2}(?=(.{2})+(?!.))|.{1,2}$/g),
                blkPrefixes = ['www', 'm', 'mobile'],
                urlParts = cd.split('.'),
                fpcd,
                ctldParts = [],
                hostParts = [],
                ctldPart = '',
                hostPart = '',
                i = 0,
                iLen = 0;
            if (!cd) return '';
            if (arrayContains(ctld, urlParts[urlParts.length -1])) {
              for (i = urlParts.length -1; i >= 0; i -= 1) {
                if ( i >= urlParts.length - 3 && urlParts[i].length <= 4) {
                  ctldParts.push(urlParts[i]);
                } else {
                  hostParts.push(urlParts[i]);
                  break;
                }
              }
              ctldParts = ctldParts.reverse();
              for (i = 0, iLen = ctldParts.length;i < iLen; i += 1) {
                if (!arrayContains(blkPrefixes, ctldParts[i])) {
                  ctldPart += i < iLen ? '.' + ctldParts[i] :  ctldParts[i];
                }
              }
              hostParts = hostParts.reverse();
              hostPart = hostParts[hostParts.length - 1] || '';
              if (arrayContains(blkPrefixes, hostPart)) {
                hostPart = '';
              }
            } else {
              hostPart = urlParts
              .slice(urlParts.length - 2, urlParts.length)
              .join('.') || '';
            }
            fpcd = hostPart + ctldPart;
            if (fpcd && fpcd.length > 4 && fpcd.split('.').length > 1) {
              // RFC 2109
              return 'domain=' + (fpcd[0] === '.' ? fpcd : (fpcd ? '.' + fpcd : '')) + ';';
            }
            return '';
          }
    
          function updateFirstPartyCookie(cookievalue) {
            var domain = getFpcd(location.hostname);
            var expireValue = cookievalue.split(":")[1];
            var events = parseInt(cookievalue.split(":")[4]) + 1;
            var expireDate = new Date(new Date().setTime(expireValue));
            var now = new Date();
            var site = (result.st) ? result.st : "nosite";
            var code = (result.cp) ? result.cp : (result.np) ? result.np : (result.fp) ? result.fp : "nocode";
            var evnt = (result.ev) ? result.ev : "noevent";
            var cookval = cookievalue.split(":").slice(0,4).join(":") + ":" + events + ":" + site + ":" + code + ":" + evnt + ":" + now.getTime().toString();
            cookval = cookval + ":" + hash(cookval);
            document.cookie = cookieName + "=" + cookval + ";expires=" + expireDate.toUTCString() + ";" + domain + ";path=/;";
          }
    
          function setFirstPartyCookie(expire) {
            if (!expire) {
              expire = expireDays()*24*60*60*1000;
            }
            var domain = getFpcd(location.hostname);
            var expireDate = new Date(new Date().setTime(new Date().getTime()+expire));
            var setDate = new Date();
            var identifier;
            var site = (result.st) ? result.st : "nosite";
            var code = (result.cp) ? result.cp : (result.np) ? result.np : (result.fp) ? result.fp : "nocode";
            var evnt = (result.ev) ? result.ev : "noevent";
            if (result.hasOwnProperty("i2")) {
              identifier = result.i2;
            } else {
              identifier = getUniqueID();
            }
            var cookreturnval = identifier + ":" + expireDate.getTime().toString() + ":" + setDate.getTime().toString() + ":" + domain.replace("domain=", "").replace(";", "") + ":1:" + site + ":" + code + ":" + evnt + ":" +  setDate.getTime().toString();
            var cookval = identifier + ":" + expireDate.getTime().toString() + ":" + setDate.getTime().toString() + ":" + domain.replace("domain=", "").replace(";", "") + ":2:" + site + ":" + code + ":" + evnt + ":" +  setDate.getTime().toString();
            cookval = cookval + ":" + hash(cookval);
            document.cookie = cookieName + "=" + cookval + ";expires=" + expireDate.toUTCString() + ";" + domain + ";path=/;";
            if (!checkFirstPartyCookie()) {
              // cookie not found, try it without domain
              document.cookie = cookieName + "=" + cookval + ";expires=" + expireDate.toUTCString() + ";path=/;";
              result.er = "N25";
              if (!checkFirstPartyCookie()) {
                result.er = "N26";
                return "nocookie";
              }
            }
            return cookreturnval;
          }
    
          function createCORSRequest(method, url) {
            var xdhreq = new XMLHttpRequest();
            if ("withCredentials" in xdhreq) {
              xdhreq.open(method, url, true);
              xdhreq.withCredentials = true;
            } else if (typeof XDomainRequest != "undefined") {
              xdhreq = new XDomainRequest();
              xdhreq.open(method, url);
            } else {
              xdhreq = null;
            }
            return xdhreq;
          }
    
          function setOptout(expire) {
            if (!expire) {
              // Year(s)*Days*Hours*Minutes*Seconds*1000
              expire = 1*24*60*60*1000;
            }
            var domain = getFpcd(location.hostname);
            var expireDate = new Date(new Date().setTime(new Date().getTime()+expire));
            document.cookie = OptoutCookieName + "=stop;expires=" + expireDate.toUTCString() + ";" + domain + ";path=/;";
            // delete 1st-Party-Cookie
            setFirstPartyCookie(2000);
          }
    
          function checkOptoutCookie() {
            try {
              var cookie = document.cookie.split(";");
              for (var i = 0; i < cookie.length; i++) {
                if (cookie[i].match(OptoutCookieName + "=.*")) {
                  result.ps = "out";
                  return true;
                }
              }
              return false;
            } catch(e) {
              return false;
            }
          }
    
          function delOptout() {
            setOptout(2000);
            // delete 1st-Party-Cookie
            setFirstPartyCookie(2000);
          }
    
          function getPlus() {
            if (typeof localStorage === 'object' && typeof localStorage.getItem === 'function') {
              if (localStorage.getItem("ioamplusdata") !== null && localStorage.getItem("ioamplusttl") !== null) {
                var currentDate = new Date();
                var now = currentDate.getTime();
                currentDate.setTime(localStorage.getItem("ioamplusttl"));
                if (now <= currentDate.getTime()) {
                  return true;
                }
              }
              var checkForSocio = 'https:' + '//' + ioplusurl + '/soziodata2.php?sc=' + socioToken + '&st=' + result.st + '&id=' + result.id;
              var XHR = createCORSRequest('GET', checkForSocio);
              if (XHR) {
                XHR.onload = function() {
                  var response = XHR.responseText;
                  var blockedUntilDate = new Date();
                  try {
                    if ((response.split(":")[1].split(",")[0]) == "0") {
                      blockedUntilDate.setTime(blockedUntilDate.getTime() + lsottlmin);
                      localStorage.setItem("ioamplusttl", blockedUntilDate.getTime().toString());
                      if (localStorage.getItem("ioamplusdata") == null) {
                        localStorage.setItem("ioamplusdata", response);
                      }
                    } else {
                      blockedUntilDate.setTime(blockedUntilDate.getTime() + lsottl);
                      localStorage.setItem("ioamplusdata", response);
                      localStorage.setItem("ioamplusttl", blockedUntilDate.getTime().toString());
                    }
                  } catch(e) {
                    // pass
                  }
                };
                XHR.send();
                return true;
              }
            }
            return false;
          }
          return {
            count: count,
            c: count,
            i: init,
            init: init,
            e: event,
            event: event,
            h: hybrid,
            hybrid: hybrid,
            setMultiIdentifier: setMultiIdentifier,
            smi: setMultiIdentifier,
            oi: optin,
            optin: optin,
            setoptout: setOptout,
            soo: setOptout,
            deloptout: delOptout,
            doo: delOptout,
            getInvitation: getInvitation,
            gi: getInvitation,
            getPlus: getPlus,
            gp: getPlus,
            consent: setConsent,
            ct: setConsent
          };
        })();
      }
    })(window)
