# The Director’s Signature: Stylometry of Theater Choreography via Pose and Action Estimation

<!-- .slide: data-background-video="assets/vis_fondly_phalp_coco.mp4" -->
<!-- .slide: data-background-size="contain" -->
<!-- .slide: data-background-video-loop -->
<!-- .slide: class="main-title" -->


---


## Introductions

<div class="headshots">

![Michael Rau](assets/people/michael_rau.jpeg) Michael Rau

![Peter Broadwell](assets/people/peter_broadwell.jpeg) Peter Broadwell

![Simon Wiles](assets/people/simon_wiles.jpeg) Simon Wiles

![Vijoy Abraham](assets/people/vijoy_abraham.jpeg) Vijoy Abraham

</div>

<div class="logos">

![TAPS](assets/logos/taps.png)
![CIDR](assets/logos/CIDR_on_dark.1237x677.png)
![SUL](assets/logos/sul_white.png)

</div>

:::

Welcome everyone, and thank you for joining us today. 
Our project draws from the field of cultural analytics, whereby computational methods are applied to the study of cultural heritage. We will discuss our exploratory work in bringing Machine Learning techniques to the study of theater performance and, in particular, to the study of directorial style.

The project, titled **Machine Intelligence for Motion Exegesis, or MIME**, is the result of a collaboration between Assistant Professor Michael Rau, and the Developer Team at Research Data Services. That team consists of Vijoy Abraham, Peter Broadwell and Simon Wiles.

**Michael Rau** is Assistant Professor of Theater and Performance Studies as well as an ~affiliate faculty member~ at HAI here at Stanford University. He received an MFA in Directing from Columbia University and has directed works **internationally** including plays, operas, and digital media projects, and has been featured in the New York Times, the Guardian and the Telegraph. He was the recipient of a 2021 **Artists + Machine Intelligence** Research Award from Google.

**Peter Broadwell** is one of two Digital Scholarship Research Developers at RDS, where his work applies machine learning, web-based visualization, and other methods of digital analysis to complex cultural data. He has a Ph.D. in Musicology from UCLA and an M.S. in Computer Science from UC Berkeley.

**Simon Wiles** is the other Digital Scholarship Research Developer on the team. Before coming to Stanford in 2011 to begin a PhD in Buddhist Philology and Philosophy- he worked, studied, and taught in Asia for many years, where he further developed his extensive technical background across several areas of study.


---


## ➡️ &nbsp; The Problem: Understanding Pose in Theater <!-- .element: class="fragment custom order-of-sections" -->


## ➡️ &nbsp; Methodology: Models and Tools <!-- .element: class="fragment custom order-of-sections" -->


## ➡️ &nbsp; The MIME Platform <!-- .element: class="fragment custom order-of-sections" -->


## ➡️ &nbsp; Results and Analysis <!-- .element: class="fragment custom order-of-sections" -->


## ➡️ &nbsp; Implications and Future Directions <!-- .element: class="fragment custom order-of-sections" -->

<!-- .slide: class="order-of-sections" -->

---


# The Problem

:::
Pose and staging lies at the intersection of authorial intent, directorial vision, and is shaped by design choices and is ultimately mediated by the performer.  So examining pose and staging in theater can be challenging since it sits at the heart of artistic expression and is so common and fundamental to the theater that it is often ignored. Our research addresses a fundamental question: How can we quantify and analyze the physical arrangements and movements of actors on stage to reveal meaningful insights about the director's creative contribution?


---


## The Problem (and Opportunity)

<div class="screenshots">

![Michael Hampe](assets/results/Hampe-Karajan_nopose.png)

![Romeo Castellucci](assets/results/Castellucci-DG_nopose.png)

![Erik Söderblom](assets/results/Soderblom-Helsinki_nopose.png)

![Sven-Eric Bechtolf](assets/results/Bechtolf-Zurich_nopose.png)

![Damiano Michieletto](assets/results/Michieletto-La-Fenice_nopose.png)

![Jean-François Sivadier](assets/results/Sivadier-Aix_nopose.png)

</div>

<!-- .slide: data-transition="slide-in fade-out" -->
:::
To answer this question, we turned to computer vision algorithms capable of detecting precise poses of the actors for every single frame of video in an archival production. Our methodology involves running pose detection on numerous archival videos, which generates hundreds of thousands of poses, and then sift through that pose data in order to draw meaningful conclusions.


---


## The Problem (and Opportunity)

<div class="screenshots">

![Michael Hampe](assets/results/Hampe-Karajan.png)

![Romeo Castellucci](assets/results/Castellucci-DG.png)

![Erik Söderblom](assets/results/Soderblom-Helsinki.png)

![Sven-Eric Bechtolf](assets/results/Bechtolf-Zurich.png)

![Damiano Michieletto](assets/results/Michieletto-La-Fenice.png)

![Jean-François Sivadier](assets/results/Sivadier-Aix.png)

</div>

<!-- .slide: data-transition="fade-in slide-out" -->

:::
In traditional theater studies, the concept of pose is often taken for granted. Directors, performers and audiences intuitively understand the power of a well-crafted tableau or a precisely choreographed sequence of movements.  In the theater, certain iconic poses or choreography can define productions, like Brecht’s silent scream choreography in Mother Courage, or Bob Fosse’s shoulder roll and arm pops in The Pajama Game or the collective poses of the ensemble in "A Chorus Line.” These production’s indelible poses helped make the work memorable, and are a signature of a particular director’s contribution to the production, and serve as a shorthand for identifying a director’s style.

However, these individual, memorable poses are just the tip of the iceberg. Our work lies in understanding the aggregate effect of all poses throughout a production, and even more ambitiously, across multiple productions by the same director, or by different interpretations of the same material by different directors. This approach brings us closer  to identifying the director's contribution to pose within the theater.

In film studies, there is a somewhat dated theory, called auteur theory which provides a framework for understanding a director's unique imprint on their work. Film directors have a range of tools at their disposal – camera angles, editing techniques, lighting choices – that make their stylistic signatures more readily apparent. In theater, however, the director's expressive capabilities are more constrained, and their contribution can be more elusive to pin down. In the theater, a director is constrained more by the physical space that they’re staging in, and the fact that their work happens in real-time, by live actors,who mediate the director’s staging.

This is where the computational analysis of pose in theater becomes interesting. By leveraging technologies like pose estimation and action recognition, we can begin to quantify aspects of theatrical performance that were previously left to subjective interpretation. We can analyze not just individual poses, but patterns of movement, spatial relationships between performers, and even the rhythm and flow of a production. Moreover, we can, as Peter will later discuss, use pose analysis to distinguish between the work of different directors, regardless of the actors, performance text, or designer. In this way, we can get closer to understanding the specific contributions that a director provides. This is important because a director’s work is often occluded, sometimes not even mentioned within theater reviews, and most importantly, by several funding agencies considered, not “generative artform, but rather an interpretative artform” and thus not worthy of funding.

The question that you might be asking yourself, is “A purely data-driven analysis might pick up on patterns and consistencies in a director's work, but can it capture the nuanced, thematic use of pose that a trained theater scholar might recognize immediately?” This is the delicate balance we must strike. On one hand, we have the potential to uncover patterns and stylistic elements that might not be apparent to the naked eye, especially when analyzing a director's body of work as a whole. On the other hand, we risk reducing the rich, complex art of theater direction to a series of data points.

Our solution lies in a synthesis of approaches. By combining the insights of traditional theater scholarship with the analytical power of computational methods, we can begin to develop a more comprehensive understanding of pose in theater. We can use data to support and enhance our qualitative analyses, and we can use our human understanding of theatrical context to guide our interpretation of the data.




---


# Methodology


---


## Temporal-Convolutional vs. Transformer Models

<div class="img-row">

![OpenPifPaf not doing well](assets/methods/OpenPifPaf_bad.jpg "OpenPifPaf not doing well") <strong>Open PifPaf</strong><br> Temporal-convolutional<br>2D only, limited tracking

![PHALP doing well](assets/methods/PHALP_WiLoR_good.png "PHALP and WiLoR doing well") <strong>PHALP</strong> (pose) + <strong>WiLoR</strong> (hands)<br>Transformers-based<br>3D, more robust to occlusion

</div>

:::
One key objective of this project has always been that it should work with standard 2D moving images taken with regular cameras, without needing any special 3D depth data or input from motion capture dots and wearables, which opens up basically the entire history of theater on film for analysis. Yet computationally extracting accurate pose data from such recordings can be quite challenging, especially if they're "in the wild" live stage or studio recordings, with multiple cameras, cuts, occlusion, shadows and so forth. In fact, this wasn't possible at all until about 2018, when advances in convolutional neural networks for computer vision led to the OpenPose project from CMU.

These convolutional approaches improved during the following 5 years, and they worked well enough on recordings made in controlled environments and for things like TikTok videos with only one or two people who are largely in the frame at all times. But as you can see from the image on the left, they struggled with in-the-wild videos such as the example shown here from the Stanford TAPS mainstage production of Julius Caesar from 2023 (directed by the illustrious Michael Rau) and were even worse at estimating 3D coordinates and tracking multiple people in the shot.

So when we began this project, we were basically hoping that better models would come along to give us better data. And as sometimes happens during the current era of AI research, we actually able to pull off the equivalent of jumping out of a plane and having someone toss a better parachute to us. This parachute came in the form a of a new generation of transformer-based models for computer vision, and more specifically via some great software tools for pose estimation and tracking from a research group across the Bay at UC Berkeley.


---


## PHALP: Predicting Human Appearance, Location and Pose

<img class="r-stretch" src="assets/methods/phalp_teaser.png" />

Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Jitendra Malik. “Tracking People by Predicting 3D Appearance, Location & Pose.” arxiv.org/abs/2112.04477 (2021).

:::
The most crucial of these tools is charmingly named PHALP -- you can see the acronym there -- and it achieves what is still basically state-of-the-art accuracy in pose estimation by interweaving the tasks of estimating human forms while also noting their appearance (that is, by extracting texture maps of their clothing and such) and tracking their trajectories over time in an estimated 3D space.


---


## LART: Lagrangian Action Recognition with Tracking

<img class="r-stretch" src="assets/methods/LART.png" />

Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Christoph Feichtenhofer, Jitendra Malik. “On the Benefits of 3D Pose and Tracking for Human Action Recognition.” arxiv.org/abs/2304.01199 (2023).

:::
The 3D pose tracking abilities of PHALP gave us much better pose data to use when it is applied to theater videos. We later augmented this with more data from a customized action recognition tool the same team built, named LART (they really like these one-syllable acronyms). The "Lagrangian" bit means that rather than taking the approach of other tools and only describing what's happening in the entire scene, LART considers each pose tracklet separately, producing distinct action descriptions (from a taxonomy of "atomic visual actions") for each person in a shot. Crucially, this software also produces a 60-element vector to describe every detected action, which provide much more computationally meaningful descriptions of the actions than the simple labels from the taxonomy (things like "watch, stand, walk").


---


## Probabilistic View-Invariant (2D+) Pose Embeddings

<img class="r-stretch" src="assets/methods/view-invariant_embeddings.jpg" />

Sun, Jennifer J, Jiaping Zhao, Liang-Chieh Chen, Florian Schroff, Hartwig Adam and Ting Liu. “View-Invariant Probabilistic Embedding for Human Pose.” In Proceedings of the European Conference on Computer Vision, Springer, 2020, pp. 53-70.

:::
The final tool that we adopted to extract, analyze and compare pose data does not come from Berkeley, rather this is from Google and Caltech. It provides a trained model that can project the 2D coordinates of a pose into a probabilistic embedding space (which actually has 16 dimensions), that situates poses closer together if they are probably similar in their actual 3D representations. Note that we also get the estimated 3D coordinates of the poses from PHALP, but as we'll see later, sometimes this probabilistic embedding is more useful for analysis.


---


# The MIME Platform


:::
* To allow us to experiment and work with these techniques and models, we have built a platform to support our development and to allow us to make this work accessible to Prof. Rau and to other researchers

* it's really the platform that makes it easy for us to construct pipelines for processing and running inference against videos of theatrical performances.

* it's a place where we can experiment with and evaluate different approaches and technologies, and we're also able to incorporate new developments as they arise, for example the shift from OpenPifPaf (the convolutional model) to the transformer-based PHALP model.

* Furthermore the platform allow us to interrogate the results using things like
  * similarity metrics
  * nearest neighbour search
  * clustering,
  * and to visualize and explore the various data we're able to produce.

---


## The MIME Platform

<section>
<div>

![Platform Diagram](assets/building/platform_diagram.svg)
<svg class="spotlight" viewBox="0 0 1080 300" preserveAspectRatio="none">
  <script type="application/json" class="spotlight-data">
  [
    {"x":"17px","y":"28px","height":"215px","width":"245px"},
    {"x":"295px","y":"8px","height":"143px","width":"459px"},
    {"x":"351px","y":"155px","height":"125px","width":"215px"},
    {"x":"629px","y":"155px","height":"125px","width":"150px"},
    {"x":"828px","y":"131px","height":"168px","width":"251px"},
    {"x":"772px","y":"18px","height":"92px","width":"295px"}
  ]
  </script>
  <defs>
    <mask id="spotlight-mask">
      <rect width="1080" height="300" fill="white" opacity=".6"/>
      <rect x="0" y="0" width="1080" height="300" fill="black"/>
    </mask>
  </defs>
  <rect width="1080" height="300" fill="#000" mask="url(#spotlight-mask)"/>
</svg>

</div>

:::

This is a diagram of the platform, more-or-less as it exists now
* it's somewhat loosely-coupled, and this is part of what allows us to easily swap in and out different parts


* The hub of the platform is the vector database where the results of the lengthy and expensive inference and computational tasks are stored for retrieval and analysis

  * we did look at dedicated vector database servers, things like Pinecone, Weviate, Qdrant etc, but even though it was fairly new at the time we decided to go with the `pgvector` extension for the PostgreSQL database engine, and that's a choice we'd make again
    * in addition to all the benefits of PostgresSQL (including the fact that we were already comfortable with it), `pgvector` has been very solid
    * it offers an array of in-engine similarity metrics and approaches to ANN-based indices (IVFFlat / HNSW)
    * and performance has been fine
      * the numbers change as we add and remove performances from the corpus,
      * we currently have around 20 million embeddings in the database,
      * and at this scale vector search is never a bottle-neck for us (although we do need to regularly retune the ANN indexes).

* The application and inference server is container that has all our machine-learning dependencies available (so, CUDA and so on, as well as OpenCV and ffmpeg etc.) and its where all our machine learning and back-end  code runs
  * the expermental nature of the project is such that this is a big ol' container; it provides a full data-science and machine-learning stack including the tensorflow and keras libraries, *as well* as pytorch, the scikit-learn stack, deepface, etc. and also the more domain-specific dependencies and toolchains mentioned earlier.
  * the container also runs a Jupyter Notebook server that I'll mention again later, and a FastAPI server that exposes the endpoints that are consumed by our web interface

* Our web-ui is internal-facing (the platform is not exposed publicly at all at this time), and it's built with Svelte components on top of  the Astro framework.  We've learned a lot through working with the interface by this point, how best to present and interact with the data, and we're currently developing a mk.2 version which is using Sveltekit.

* The web-ui and notebook server is sat behind a reverse proxy that makes it available to web-browsers

* The whole thing is orchestrated with docker which makes it
  * relatively easy to deploy across infrastructure platforms
  * and allows us the same containerized environment across the inference and experimentation cycles (and we can also do training in that environment)
  * and we can just deploy and fire up a notebook and get hacking against a replica of the production environment


* so let's take a look at the interface

---


## Platform Demonstration

<video id="platform-demo-video" controls muted src="assets/mime-hai-seminar-demo-video.mp4"></video>

:::

# NOTE: Probably the demo should be shorter, and focus on the aspects related to pose comparison
# and bulk analysis for stylometry. Maybe spend more time showing the mk2 interface?

* Performances
  * On the initial screen we can see  the table of recordings that have been ingested into the platform
    * The corpus of performances that we have in the system varies as we add and remove things according to our current interest; at present we have 11 recordings and a total of about 7 million poses.
    * "Tracks" here refers to the number of appearances of a figure (an actor) who can be tracked across frames
      * so for Julius Caesar here we have well over half a million individual poses detects, and this maps to a little under 3,000 tracks, so 3,000 unique sequences of the same figure moving throuh the same camera shot.
    * "Shots" is the count of distinct camera shots; when a there's a scene change or a camera shot change, this number is incremented

* So let's select the Stanford TAPS' production of Julius Caesar from March 2023, which was directed by Prof. Rau, and take a look.

* Timeline
  * The primary interface to the generated dataset is oriented around a timeline chart
  * Here we have a time series along the x-axis, and a number of metrics of interest are plotted against the y-axis
  * In the first place we have the track count, this green line, which represents the number of actors detected in the frame
    * the face counts follow the track counts quite closely, and we would hope, and these discrepancies are mostly where actors are in the frame but are facing away from the camera
    * here at the end of the production we can find the curtain call, where of course we have the highest track count and face count, as the whole cast is on stage and facing the camera
  * the average score metric is the confidence score provided by the pose prediction model
  * the pink lines represent camera shot changes  
  * the MIME platform also derives metrics that represent the movement of figures on the stage, as well as indices of what we are terming pose and action "interest" -- the degree to which the poses and actions in the frame differ from baseline or "typical" poses and actions across the recording as a whole


  * we can zoom in to this five-to-ten minute section here that covers the assasination of caesar
    * frame # ~95,733 shows the conspirators standing around the body of Caesar
    * we can see that the pipeline has done a decent job of picking out the seven actors on the stage in the this frame, including Caesar on the floor

    * let's focus on Cassius who has their back to us here and is walking towards Caesar's body
      * and search for similar poses across the performance
    * on the left we have a 3D rendering of the pose itself (and we have an experimental editor we've been working on)
    * and a card representing the source pose itself
    * and then here we can see the top matches across the recording
      * these small cutouts are a little difficult to see, especially when there's minimal contrast in the footage
      * we can draw the pose diagrams over the top of the images to make them a little easier to see what's going on
        * and now we can see we've found a number of other parts in the performance where an actor has their back to us and is walking away
      * we can also return to the timeline to get a quick overview of where the most similar poses occur in the context of the recording as a whole
      * ---

    * the platform also allows us to work with and evaluate different similarity metrics for our searches
      * here we're searching the view invariant embeddings instead of the original embeddings
      * and this allows us to find people in similar walking poses irrespective of their orientation with respect the stage or the camera,
      * we can see here we have found more parts of the performance where we have actors in similar walking poses but in different directions and from different angles
    * ---
    * in addition to the pose similarity search, we can also search for similar actions, using data derived using the LART technique
      * we're still working on this, but it's interesting to see that different matches are surfaced when the additional movement context is part of the equation

  * 3d scene
    * using the "3D scene" toggle here we can activate the 3d reconstruction of the frame
    * the 3d predictions provided by the pipeline allow the platform to present a visualization that represents the positions of the actors with respect to one another
    * these data can support lots of different kinds of analyses, some of which will be discussed later
    * here we can see Caesar on the floor with Cimber, Brutus is at the top of the steps, and here's Cassius in the pose we were just searching for

  * so these are some of the affordances we have in the timeline view, and we can take a look now at some of the other views

* Faces
  * The "faces" interface is  a dot plot timeline that presents clusters that have been identified by face recognition and shows when recognized faces appear during the performance.
  * this is very experimental still; we were hoping we could use face recognition to augment the actor-tracking, with the primary goal of tying together tracks that correspond to the same actor across shots
  * this is still something we're working on, but the results here aren't as good as we'd like, mainly due to the very inconsistent quality of face recognition on footage like this
  * so were looking at other ways to achieve our goals here

* Poses
  * the pose  cluster visualization presents a similar chart, shown here are 15 **groups** of poses that have been clustered on the basis of a similarity metric and they're shown as they're distributed across the timeline of the performance
    * in addition to using different similarity metrics, the clusters themselves can be produced using a variety of techniques;
    * here we've just used a simple UMAP algorithm for dimensionality reduction and the HDBSCAN algorithm to perform the clustering, but there are many other possibilities
  * so for example cluster 15 is a cluster of seated poses including this section where MIchael has Brutus sitting on the front of the stage
  * and we can see how poses in that similarity cluster are distributed across the performance

* Explorer
  * Another way of exploring the clusters is with the "PosePlot" explorer, which some folks might recognize as a modification we made of the PixPlot explorer from the Yale DH lab
  * the cluster exemplars are over here on the left, and we can see our # 15 cluster of sitting folks over here,
    * here we have a little group of Lucius sitting with hands on knees, leaning a little forward
    * whereas over here we have a group from the same scene where they're leaning back a little more with their hands more in their lap
    *
  * there's also, for example, and interesting cluster down here (#14) of folks standing with their right arms down and their left arms bent, perhaps in a pocket, opening their bodies to the audience (here we have mainly Cassius, Casca, and Marcus Antonius I think)
  * the PosePlot explorer also allows a timeline-like view which we sometimes refer to as the skyline view, where the poses are binned and stacked by minute, and this allows us another way to get a sense of how different clusters of poses are distributed across the performance


* okay, so that's some of what we have in our mk.1 interface

* I do also want to quickly show the mk.2 interface and how we're building on and improving what we've got
  * here's the search interface as we see matches for that same pose we were looking at earlier
    * this presentation makes it much easier to see the actors, even if the contrast is still sometimes an issue
    * we can still view the pose overlay, of course
    * and it's easy to get a view of the whole frame for additional context

  * searching by all the available similarity metrics is still supported, here we're searching the view invariant embeddings again

  * but one of the main focuses of the new interface is make it easier to operate across multiple performances
    * so here you can see we're searching for that same pose performed by Cassius in Julius Caesar, but now we're returning matches from across the entire corpus of productions available to the platform
    * and this opens up new possibilities for exploration
  
  * One thing that started out as a bit of fun but has proved surprisingly useful is the search-by-webcam feature we developed;
    * here a frame can be captured from a webcam, then inference is performed client-side in the users' browser to produce a pose vector in the form we need, and then we can search the MIME database for matches
      * so here I've performed a pose for the camera and we can look for matches in Julius Caesar, including these wonderful shots of a red-handed Brutus
      * and we can also look for similar poses across the


* And last but not least, the mk.1 interface provides a direct link to the embedded Jupyter server.
* This has been really useful to us because it allows us to very easily spin up notebooks that are running remotely in the same containerized environment where our inference and analysis is done;
  * same hardware, same dependencies installed and available, same interface to our embedding databases etc.
* and this is extremely convenient for prototyping, experimenting, and just noodling around, as well as for conducting more formal analyses.  Peter is going to share some of the early results of some of those analyses now.

---


# Results & Analysis


---


## The Test Corpus: Assembling Multiple Works per Director

<img class="r-stretch" src="assets/results/31_performances.png" />

:::
To approach the question of what the pose, action, motion and position data we can extract from recordings via the MIME platform might tell us about how theater directors deploy these elements to produce specific effects, while imprinting their own stylistic signature, we focused on three high-profile, contemporary "auteur" directors: Bill T. Jones, Romeo Castellucci, and Krzysztof Warlikowski, selecting 10 or 11 performances each has directed during his career. We used commercial or archival recordings of these performances, running them through the MIME pipeline without alteration, other than upscaling some of them to at least HD quality, because the models work better with high-resolution footage.

As you can see, this produced 10-20 hours of pose data per director. The table shows many of the summary statistics of the performances, which we'll discuss in more detail soon.


---


## Which Features Are Best for Differentiating between Directors?

<div class="img-row">

Pose motion and distance statistics
![Pose motion and distance confusion matrix](assets/results/LOO_motion_distance.png "Pose motion and distance confusion matrix")

View-invariant pose embeddings
![View-invariant pose embeddings confusion matrix](assets/results/LOO_POEM_features.png "View-invariant pose embeddings confusion matrix")

</div>

:::
We framed our computational approach to directorial style as a performance-to-director classification problem: from the derived pose, action, motion and distance data, which elements are most effective at predicting the correct stage director for a given performance? We began with “leave one out” (LOO) tests in which we assembled the average pose distance and motion features into a vector for each performance, then did the same for the view-invariant pose embedding vectors and action recognition vectors, and also the 3D “global orientation” pose keypoint coordinates. We then compared these to aggregate vector averages of these values across all of a director’s works except the held-out performance, matching a performance's average vector to the director's vector to which it has the highest cosine similarity.

Here you can see the resulting confusion matrices from some of these simple "leave one out" experiments. The basic pose and motion statistics were not as effective at differentiating works by Castellucci and Warlikowski as they were at telling the difference between Bill T. Jones and the other two, while the aggregated pose embeddings were more successful at differentiating the works of all three.


---


## Which Features Are Best for Differentiating between Directors?

<img class="r-stretch" src="assets/results/31_performances_hl.png" />

:::
We've returned briefly to the full table of performances, now with some colorization to point out recordings with higher in-place or "sidereal" (relative to the backdrop) motion and inter-pose distances. As you can see, most are from Bill T. Jones, who primarily directs dance works, while the other two tend to direct mostly operas and other more static forms of stage plays. So the motion and distance-based comparison may be better at clueing in to the difference in primary genres between the directors, while the pose embedding approach seems to be picking up something distinctive about each director's style. We'll look into what that might be in a minute.


---


## Which Features Are Best for Differentiating between Directors?

<div class="img-row">

Body keypoint coords (3D)
![3D coordinates confusion matrix](assets/results/LOO_3D_coords.png "3D coordinates confusion matrix")

Action recognition embeddings
![Action recognition embeddings confusion matrix](assets/results/LOO_action_features.png "Action recognition embeddings confusion matrix")

</div>

:::
Here are some further leave-one-out classification experiments with the 3D pose coordinates and action recognition vectors used as features. The action recognition embeddings do seem to be somewhat better than the 3D coordinates for telling the directors apart.


---


## Which Features Are Best for Differentiating between Directors?

<img class="r-stretch" src="assets/results/classification_experiments.png" />

:::
The results of the leave-one-out tests are largely replicated in analyses involving more sophisticated classification algorithms and using 10-fold cross-validation with different random seeds, when available, to try to get a better sense of how these approaches might do with a larger, more diverse dataset. Remember that 33% accuracy would be expected by chance. It's interesting, however, that the Gaussian Naive Bayes classifier is nearly as good, and certainly within the margin of error, at differentiating between directors based on motion and distance statistics as the formerly dominant embedding approaches are with any approach. (Note that we tried many other classifiers here, including some fancy neural models, but their results were all in the ranges of the Random Forest and Gaussian Naive Bayes methods).

---


## Which Features Are Best for Differentiating between Directors?

<div class="img-row">

Pose motion and distance
![Pose motion and distance feature importances](assets/results/importances_motion_distance_nb.png "Pose motion and distance feature importances")

View-invariant pose embedding
![Pose embedding importances](assets/results/importances_poem_nb.png "Pose embedding importances")

</div>

:::
To get at the question of exactly which elements of these feature sets the models found the most useful at differentiating between directors, we ran some feature importance tests (specifically with the Gaussian Naive Bayes model, as it generally performed the best). Although the error bars are large (again) due to the small set of performances to be classified, it seems that the average amount of motion in a video is one of the most powerful predictors of the director's identity, while the average distance between performers on stage is not so useful.

In the case of the embeddings, however, it's more difficult to determine what the salience of "feature 4" (out of 16) actually means for poses -- we'll show some initial steps we've taken to dig into this in a couple of slides.


---


## Which Features Are Best for Differentiating between Directors?

Body keypoint coords (3D)  
<img class="r-stretch" src="assets/results/importances_keypoints_nb.png" />

:::
And considering the 3D body keypoint coordinates, this plot might be a bit hard to read, but essentially the lateral positions of the right wrist, and then the left wrist as viewed from the "global" perspective (essentially from directly in front of the actor) was the most indicative of a director's style, followed by the lateral apsects of the right elbow and knee, then other aspects of the shoulders, knees, ankles, nose, with the hips being less important. The data are quite noisy here, but interestingly the right-handedness bias (due to up to 90% of the global population being right-handed) does seem to extend to the right side of the body more generally.

---


## Visualizing Directors' Pose "Repertoires"

<img class="r-stretch" src="assets/results/poem_umap_sampled_2.png" />

:::
To explore how the directors' pose "repertoires" might be distributed through the view-invariant pose embedding space, and hopefully get a better sense of what aspects of these embeddings were helping to differentiate the directors, we plotted a 1% sample of all of their pose embeddings into a 2D space via UMAP projection as color-coded dots. The larger hexagons represent each director's overall average pose embedding. Although we can't directly translate these averages into poses, we can search for the most similar poses in the entire data set thanks to MIME's powerful vector database, and we've shown some of those here in the callouts.

This projection does seem to reveal useful insights, like the overall greater similarity between Castellucci’s and Warlikowski’s pose repertoires relative to Jones’s, and the callouts highlight how Jones tends to employ dramatically contorted poses, Castellucci to favor hunched-over standing postures, and Warlikowski to deploy figures in an active sitting position.


---


## Direct Comparison: Multiple Directors' Stagings of the Same Work

<div>

![Comparisons of performances of Don Giovanni](assets/results/dg_comparison_table.png "Comparisons of performances of Don Giovanni")

</div>

<div class="screenshots">

![Michael Hampe](assets/results/Hampe-Karajan.png) Michael Hampe

![Romeo Castellucci](assets/results/Castellucci-DG.png) Romeo Castellucci

![Erik Söderblom](assets/results/Soderblom-Helsinki.png) Erik Söderblom

</div>

:::

# NOTE: Not sure whether to include this section...
# If we have more results involving hand analysis and archetype "fingerprints" 
# (Delsart or otherwise), we probably should show those instead

Our final analytical experiment to date, which is still a work in progress, sort of inverts the previous approach and instead considers what MIME can reveal when comparing staging from seven different directors who are all directing the same work -- in this case the famous Mozart/Da Ponte opera Don Giovanni from 1787.

The screenshots below (which you may recall from the beginning of the seminar) show some pose estimation output from the directors -- note that in every case this is the same scene, from the finale of Act I.

---


## Direct Comparison: Multiple Directors' Stagings of the Same Work

<div>

![Comparisons of performances of Don Giovanni](assets/results/dg_comparison_table.png "Comparisons of performances of Don Giovanni")

</div>

<div class="screenshots">

![Sven-Eric Bechtolf](assets/results/Bechtolf-Zurich.png) Sven-Eric Bechtolf

![Damiano Michieletto](assets/results/Michieletto-La-Fenice.png) Damiano Michieletto

![Jean-François Sivadier](assets/results/Sivadier-Aix.png) Jean-François Sivadier

</div>

:::
The table above gives numerous derived statistics of the performances, colorized to highlight entries with greater degrees of movement, distance between actors, and overall deviation from the average pose and actions (aka "interest" as we saw in Simon's demo of the MIME interface). As the table is also ordered chronologically, it does seem to indicate a general tendency for performances to get more creative and/or "non-traditional" in their stagings over time, as quantified via pose and action estimation.


---


## Aligning Performances (by Music) to Get Average Pose "Consensus"

<div class="r-stack">
  <img
    src="assets/methods/Don_Giovanni-Soderblom-Helsinki_UzxYEVbOS5w.mp4_chroma_time_correspondences.png"
  />
  <img
    class="fragment"
    src="assets/methods/Don_Giovanni-Soderblom-Helsinki_UzxYEVbOS5w.mp4_chroma_comparison.png"
  />
  <img
    class="fragment"
    src="assets/methods/Don_Giovanni-Soderblom-Helsinki_UzxYEVbOS5w.mp4_chroma_warping_path.png"
  />
</div>


:::
Comparing the performances of Don Giovanni involved aligning all of them down to the sub-second level, specifically so that we could calculate the average poses and actions deployed at each moment of the opera -- building what folklorists refer to as the "consensus performance" or "tradition dominant" of a cultural expressive form -- and then calculate the degree to which each director's staging deviates from this consensus at each moment.

Practically speaking, the easiest way to align the recordings, which only works because the work is an opera, involves using some old-fashioned music "AI" techniques, extracting the musical pitches heard at each timecode, then applying the dynamic time warping algorithm to match the pitch chroma of different performances to compute a warping path that allows us to align them despite significant differences in tempo, non-musical action, and the occasional omission of certain scenes.


---


## Comparing Each Director's Staging to the "Consensus" Average

<div class="r-stack">
  <img
    src="assets/results/all_movement3d.png"
  />
  <img
    class="fragment"
    src="assets/results/consensus_all_movement3d.png"
  />
</div>

:::
As a final analytical output of the effort just described, we can plot the pose or motion similarity attributes of each of the 7 performances to the average  "consensus" performance across the entire work (the dashed lines are scene and act boundaries). This is still a work in progress, but we can use this analysis to detect patterns such as certain scenes in which stagings are more likely to deviate from the consensus poses, and eventually use this to highlight automatically where directors might use especially distinctive poses and actions. Stay tuned...


---


## Comparing Each Director's Poses to the "Consensus" Average

<section data-background-iframe="assets/bokeh/dg_poem_comparison.html"
         data-background-interactive>
</section>


---


# Implications

:::
The computational analysis of pose and action in theatrical performances, as presented in this research, opens new avenues for understanding directorial style. This is particularly significant because the creative contributions of directors are often overlooked or reduced to a single memorable moment or tableau. In reality, the staging of a theatrical production is a meticulous process that unfolds over weeks or even months, with the director crafting every moment of the performance. By leveraging this technology to examine their work in minute detail, we gain a more comprehensive view of their artistic vision.
And from my position, this technology is opening whole new ways of examining a performance, so I’m going to just go over some of the ways in which pose can allow for a unique analyses of a performance.  By using the pose similarity function we can start to identify recurring poses, symmetry within poses, and looking at the overall timeline view of a performance, to see the rhythmic ebb and flow of a staging. Using pose we can more readily identify the common themes and stylistic elements that define a director’s creative output, and separate out their unique contribution, divorced from the work of the performers, or the constraints of the particular text or physical space.
By analyzing pose and movement across a director’s entire body of work, we can identify recurring patterns and trace the evolution of their style—elements that might remain elusive when examining individual performances in isolation. Additionally, this method enables objective comparisons between different directors’ interpretations of the same material, potentially revealing new dimensions of artistic expression and decision-making. By comparing different director’s versions of the same work, we can start to see how physical expression, and spatial storytelling evolves over time, or across different cultures, or in specific canonical works, look at how a director’s staging deviates from the traditional norms. Scholars could look at certain themes or characters that are represented physically over time, an easy topic would be to look at how kings are represented in different stagings, to see how poses associated with power and submission can vary over time. This technology could also be used to chart how an actor’s physical choices evolve through rehearsals and performances. Then scholars might use the rehearsal and performance pose data to study methodologies of actor training or directorial vision. Given the precision of the timeline view, we can also correlate pose data with audience response (based on applause or laughter) to see which types of physical expression elicit the strongest responses, so that could deepen our understanding of how audiences engage with a live performance. As a practicing artist, I see this tool enabling new types of staging, helping to rethink pose within canonical performances, and inspiring innovative uses of pose. And ultimately I see the potential of this technology to create an unprecedented record of a production’s physical language, which can be a valuable resource for future research, teaching, or restaging. The potential for scholarship and analysis enabled by this technology is vast.
Moreover, the implications of this research extend beyond theater studies. First, this methodology can be readily adapted to analyze performances in film, opera, dance, and other any of the arts involving human movement. Beyond the arts, this approach could also be applied to diverse fields, from analyzing political speeches to studying the biomechanics of physiology and sports.
However, it is crucial to acknowledge the ethical considerations and limitations inherent in this approach. As with all AI, there is a danger in hallucinations and incorrect pose estimations. There are always responsible use considerations when working with archival materials of directors, actors. This presents a complex ethical landscape that requires careful navigation, and as a team we’ve developed clear guidelines and frameworks to ensure respectful and responsible use of these technologies and materials.
From an analytical standpoint, we must also recognize the limitations of focusing solely on pose and action. Theater is a multifaceted art form, and while pose is a critical component, it represents only one thread in a rich tapestry that includes dialogue, set design, lighting, sound, and more. Without taking those other elements into account, we run the risk of not interpreting a production properly. While this computational analysis provides valuable insights, it should be viewed as a complement to, rather than a replacement for, traditional methods of artistic analysis. It offers a narrow but powerful lens through which to examine performance, enriching our understanding rather than supplanting it.
Finally, it is important to note that this computational approach cannot definitively determine directorial intent. The patterns observed may stem from conscious directorial choices, actor improvisations, or even unintentional recurring elements. As such, these findings should serve as a starting point for deeper, more nuanced investigations rather than as definitive conclusions.
In conclusion, while this research presents exciting possibilities for quantitative analysis in the arts, its true value lies in its potential to complement and enhance traditional scholarly approaches. By integrating computational methods with expert human interpretation, we can develop a more comprehensive understanding of theatrical performance and directorial style, ultimately enriching both academic discourse and artistic practice.
Thank you.


---


# Thank You&#x21;

<div class="logos">

![TAPS](assets/logos/taps.png)
![CIDR](assets/logos/CIDR_on_dark.1237x677.png)
![SUL](assets/logos/sul_white.png)

</div>

