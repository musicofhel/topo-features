# TopoFeatures — Narration Script
# Total: 3:00 (180s), ~449 words at 150 wpm
# Voice: ElevenLabs "Tom" (Balanced, Clean and Approachable)
# Settings: stability=0.50, similarity_boost=0.75, style=0.0, speaker_boost=true

## Act 1: The Problem (00:00 - 00:35, 87 words)

[00:00] What shape is your time series?

[00:04] You have sensor data. You extract features — mean, variance,
spectral power, entropy. These capture statistics.
But they miss something.

[00:15] Your data came from a dynamical system. And that system has
geometry. Its trajectory through state space forms a shape —
loops, spirals, folds.

[00:25] That shape is called an attractor. Its structure tells you
things about the system that no statistic will reveal.

## Act 2: From Time Series to Point Cloud (00:35 - 01:05, 75 words)

[00:35] How do we get at the shape?

[00:38] If you have multiple sensors, project down to three
dimensions with PCA. You get a point cloud.

[00:46] If you have one sensor, use delay embedding — take the value
now, ten steps ago, twenty steps ago. Each triple is a point.
Stack them up and you reconstruct the attractor from a single
channel.

[00:58] Either way: a point cloud with a shape we can measure.

## Act 3: Growing Balls (01:05 - 01:55, 125 words)

[01:05] Here's the key idea. Put a tiny ball around each point.
Now grow them.

[01:12] At first, nothing connects. Many isolated clusters.
We count those — that's beta-zero.

[01:20] Keep growing. A ring of points connects into a loop — with
empty space inside. Beta-one just ticked to one. A topological
feature is born.

[01:33] Grow further. The loop fills in and dies. It was born at
one scale and died at another. The lifetime is the difference.

[01:43] Long-lived features are real structure. Short-lived features
are noise. That's the core insight — persistence separates
signal from accident.

## Act 4: The Persistence Diagram (01:55 - 02:25, 75 words)

[01:55] We record this in a persistence diagram. Birth on the x-axis,
death on the y-axis. Points near the diagonal — noise.
Points far from it — real structure.

[02:08] From this diagram, five numbers. Beta-zero and beta-one.
Persistence entropy. Max persistence. Total persistence.

[02:18] Five numbers. One function call.

## Act 5: What It Predicts (02:25 - 02:48, 57 words)

[02:25] These five numbers predict system properties across eighteen
out of twenty systems we tested.

[02:31] Synchronization in oscillators. Memory capacity in neural
networks. Stability in power grids. Sleep depth in real EEG.

[02:40] No domain knowledge. No model assumptions. Just the geometry
of the trajectory.

## Act 6: Use It (02:48 - 03:00, 30 words)

[02:48] Import extract. Pass your array. Get five features. Or plug it
into tsfresh alongside seven hundred standard features.

[02:56] Pip install topo-features.
