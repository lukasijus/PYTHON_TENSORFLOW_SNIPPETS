	=
ףp�L@=
ףp�L@!=
ףp�L@	E��$?P@E��$?P@!E��$?P@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$=
ףp�L@{�G�z�?AZd;�O-4@Y��C��B@*	    �X�@2P
Iterator::Model::Prefetch���x��B@!��6��X@)���x��B@1��6��X@:Preprocessing2F
Iterator::Model��K7�B@!      Y@);�O��n�?1T�If��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 65.0% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	{�G�z�?{�G�z�?!{�G�z�?      ��!       "      ��!       *      ��!       2	Zd;�O-4@Zd;�O-4@!Zd;�O-4@:      ��!       B      ��!       J	��C��B@��C��B@!��C��B@R      ��!       Z	��C��B@��C��B@!��C��B@JCPU_ONLY