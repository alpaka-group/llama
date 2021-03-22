.. include:: common.rst

.. _label-view:

Copying between views
=====================

Especially when working with hardware accelerators such as GPUs or offloading to many core procressors,
explicit copy operations call for memory chunks as big as possible to reach good throughput.

It is trivial to copy a view from one memory region to another if mapping and size are identical.
However, if the mapping differs, a direct copy of the underlying memory is wrong.
In most cases only elementwise copy operations will be possible as the memory patterns are not compatible.
There is a small class of remaining use cases where the mapping is the same, but the size of the view is different or mappings are
very related to each other. E.g. when both mappings use struct of array, but one time with, one time without padding.
In those cases an optimized copy operation would in *theory* be possible .
However *practically* it is very hard to figure out the biggest possible memory chunks to copy at compile time,
since the mappings can always depend on run time parameters.
E.g. a mapping could implement struct of array if the view is bigger than :math:`255` elements, but use array of struct for a smaller amount.

Three solutions exist for this challenge. One is to implement specializations
for specific combinations of mappings, which reflect the properties of those
mappings. This **can** be the way to go if the application shows significantly
better run times for slightly different mappings and the copy operation is the bottle neck.
However this would be the very last optimization step as for every new mapping a new specialization would be needed.

Another solution would be a run time analysis of the two views to find
contiguous memory chunks, but the overhead would probably be too big, especially
if no contiguous memory chunks could be found. At least in that case it may make
sense to use a (maybe smaller) intermediate view which connects the two worlds.

This last solution is that we have e.g. a view in memory region :math:`A`
with mapping :math:`A` and another view of the same size in memory region
:math:`B` with mapping :math:`B`. A third view in memory region :math:`A` but
with mapping :math:`B` could be used to reindex in region :math:`A` and then to
copy it as one big chunk to region :math:`B`.

When using two intermediate views in region :math:`A` and :math:`B` with the
same mapping but possibly different than in :math:`A` **and** :math:`B` the copy
problem can be split to smaller chunks of memory. It makes also sense to combine
this approach with an asynchronous workflow where reindexing, copying and
computation are overloayed as e.g. seen in the
`async copy example <https://github.com/alpaka-group/llama/blob/master/examples/asynccopy/asynccopy.cpp>`_.

Another benefit is, that the creation and copying of the intermediate view can
be analyzed and optimized by the compiler (e.g. with vector operations).
Furthermore different (sub) datum domains may be used. The above mentioned async copy
example e.g. applies a bluring kernel to an RGB-image, but may work only on
two or one channel instead of all three. Not used channels are not allocated and
especially not copied at all.
