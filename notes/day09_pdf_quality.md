# Day 09：PDF 解析质量记录

## 1. 测试概况

* 测试 PDF 数量：5
* 总页数：73
* 空文本页数量：0
* 存在解析警告的页面：8
* 有警告页面占比：8 / 73，约 11.0%

## 2. 空文本页检查

本次测试中没有出现空文本页：

```text
empty_text pages: 0
```

说明当前 5 篇测试论文的所有页面均能够通过 PyMuPDF 提取出非空文本。

需要注意，这只能说明页面中存在可提取文本，不能说明文本的阅读顺序、公式、特殊符号和版面结构均被正确恢复。

## 3. 解析警告

共发现 8 个页面包含异常字符：

### SARGAN_A_Novel_SAR_Image_Generation_Method_for_SAR_Ship_Detection_Task.pdf

* PDF page 5

  * `null_characters:6`

* PDF page 7

  * `null_characters:4`
  * `private_use_characters:10`

### Wave_Spectrum_Retrieval_Method_Based_on_Full-Link_Ocean_Surface_SAR_Imaging_Simulation.pdf

* PDF page 3

  * `private_use_characters:47`

* PDF page 4

  * `null_characters:2`
  * `private_use_characters:39`

* PDF page 10

  * `private_use_characters:9`

* PDF page 11

  * `null_characters:5`
  * `private_use_characters:22`

* PDF page 12

  * `null_characters:1`

* PDF page 18

  * `null_characters:3`

## 4. 初步判断

当前检测到的警告主要包括两类：

### `null_characters`

提取文本中包含 `\x00` NULL 字符。

这些字符通常不会出现在正常论文正文中，因此将对应页面标记为需要人工检查，但当前阶段不直接删除。

### `private_use_characters`

提取结果中出现 Unicode Private Use Area 字符。

这类字符可能与 PDF 内嵌字体、数学符号或特殊字符映射有关，因此不能简单认为页面已经损坏，也不能直接删除。

需要将警告页面与原 PDF 逐页进行人工对照。

## 5. 当前结论

本次测试的 73 个页面均能够提取出非空文本，说明基础 PDF 文本读取流程可以正常运行。

但 8 个页面出现 NULL 字符或 Private Use Area 字符，因此后续需要重点检查：

1. 异常字符是否来自数学公式；
2. 是否来自希腊字母或特殊符号；
3. 是否影响普通英文正文；
4. 是否会影响后续 Chunk 和 Embedding；
5. 是否需要针对特定页面或字符进行清理。

当前阶段只进行检测和记录，不自动删除或替换异常字符。

## 6. 双栏文本顺序测试

测试对象：典型 IEEE GRSL 双栏论文页面。

### sort=False

评价：基本正常。

正文整体保持了较合理的线性阅读顺序。上一节正文、
Contributions、Section II、Section A 和 Section B 能够按照
论文逻辑依次提取，未观察到明显的左右栏内容交叉。

存在的主要问题包括：

- 个别区域出现异常换行，例如部分句子被拆成一个单词一行；
- 存在跨行断词；
- 存在 ﬁ、ﬃ 等 ligature 字符；
- 页眉和图注等非正文信息仍包含在结果中。

总体而言，虽然局部文本仍需后续清洗，但正文语义顺序基本完整，
适合当前 RAG 文本处理流程。

### sort=True

评价：不适合当前 RAG 线性文本提取。

sort=True 能够较好地恢复原 PDF 的二维双栏空间布局，
在终端中可以看到左右两栏按照页面位置并排显示。

但在线性字符串中，同一纵向位置的左右栏内容会出现在同一文本行，
使原本属于不同段落甚至不同章节的文本相邻。因此后续直接进行
Chunk 切分时可能产生跨栏语义混合。

因此 sort=True 虽然具有较好的二维版式恢复效果，但不适合作为
当前 RAG 系统的默认纯文本提取方案。

### 当前策略

暂时采用：

page.get_text("text", sort=False)

后续继续使用其他典型 IEEE 双栏页面进行验证。
同时将异常换行、跨行断词、ligature 和页眉/页脚清理作为独立的
文本预处理问题处理，而不依赖 sort 参数解决。
