	�t�VL@�t�VL@!�t�VL@	��ըP�O@��ըP�O@!��ըP�O@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�t�VL@q=
ףp�?A�v���4@Y9��v��A@*	    `��@2P
Iterator::Model::Prefetch���K�A@!�gl��X@)���K�A@1�gl��X@:Preprocessing2F
Iterator::Model�C�l��A@!      Y@);�O��n�?1�4aN��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 63.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	q=
ףp�?q=
ףp�?!q=
ףp�?      ��!       "      ��!       *      ��!       2	�v���4@�v���4@!�v���4@:      ��!       B      ��!       J	9��v��A@9��v��A@!9��v��A@R      ��!       Z	9��v��A@9��v��A@!9��v��A@JCPU_ONLY