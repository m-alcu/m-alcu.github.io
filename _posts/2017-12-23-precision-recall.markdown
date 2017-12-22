---
layout: post
title: Precision Recall
date: 2017-12-21 9:01
comments: true
external-url:
categories: statistics
---

Let's assume that you have a cancen recognition process with 1 percent error process. But only 0.5 percent of patiens have cancer.

If you precict always 0, then you have a 0.5 percent error, that seems better than the process.

How to calculate true quality indicators?

$$
\begin{array}{c|lcr}
 & \text{Actual=1} & \text{Actual=0}\\
\hline
 \text{Predicted=1}& \color{blue}{True +} & \color{red}{False+} \\
\hline
\text{Predicted=0} & \color{purple}{False-} & \color{green}{True-} \\
\end{array}
$$

Of all patients where we predicted $y = 1$, what fraction actually has cancer?  

$$
\mathbf{Precision}= \frac{\color{blue}{True+}}{\text{# predicted positive}}=\frac{\color{blue}{True+}}{\color{blue}{True+}+\color{red}{False+}}=\frac{0}{0+0}= NaN
$$

Of all patients that actually have cancer, what fraction did we correcty detect as having cancer?

$$
\mathbf{Recall}= \frac{\color{blue}{True+}}{\text{# actual positive}}=\frac{\color{blue}{True+}}{\color{blue}{True+}+\color{purple}{False-}}=\frac{0}{0+0.5}= 0
$$