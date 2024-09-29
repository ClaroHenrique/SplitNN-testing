# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: distributed_learning.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1a\x64istributed_learning.proto\x12\x14\x64istributed_learning\"?\n\x05Query\x12\x12\n\nbatch_size\x18\x01 \x01(\x05\x12\x12\n\nrequest_id\x18\x02 \x01(\x05\x12\x0e\n\x06status\x18\x03 \x01(\x05\";\n\x06Tensor\x12\x0e\n\x06tensor\x18\x01 \x01(\x0c\x12\r\n\x05label\x18\x02 \x01(\x0c\x12\x12\n\nrequest_id\x18\x03 \x01(\x05\"\x1b\n\nModelState\x12\r\n\x05state\x18\x01 \x01(\x0c\"\x1a\n\x07Measure\x12\x0f\n\x07measure\x18\x01 \x01(\x0c\"q\n\x11TensorWithMeasure\x12,\n\x06tensor\x18\x01 \x01(\x0b\x32\x1c.distributed_learning.Tensor\x12.\n\x07measure\x18\x02 \x01(\x0b\x32\x1d.distributed_learning.Measure\"\x07\n\x05\x45mpty2\xd9\x04\n\x11\x44istributedClient\x12\x46\n\x07\x46orward\x12\x1b.distributed_learning.Query\x1a\x1c.distributed_learning.Tensor\"\x00\x12G\n\x08\x42\x61\x63kward\x12\x1c.distributed_learning.Tensor\x1a\x1b.distributed_learning.Query\"\x00\x12P\n\rGetModelState\x12\x1b.distributed_learning.Empty\x1a .distributed_learning.ModelState\"\x00\x12P\n\rSetModelState\x12 .distributed_learning.ModelState\x1a\x1b.distributed_learning.Empty\"\x00\x12T\n\x16GenerateQuantizedModel\x12\x1b.distributed_learning.Empty\x1a\x1b.distributed_learning.Empty\"\x00\x12W\n\rTestInference\x12\x1b.distributed_learning.Query\x1a\'.distributed_learning.TensorWithMeasure\"\x00\x12`\n\x16TestQuantizedInference\x12\x1b.distributed_learning.Query\x1a\'.distributed_learning.TensorWithMeasure\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'distributed_learning_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_QUERY']._serialized_start=52
  _globals['_QUERY']._serialized_end=115
  _globals['_TENSOR']._serialized_start=117
  _globals['_TENSOR']._serialized_end=176
  _globals['_MODELSTATE']._serialized_start=178
  _globals['_MODELSTATE']._serialized_end=205
  _globals['_MEASURE']._serialized_start=207
  _globals['_MEASURE']._serialized_end=233
  _globals['_TENSORWITHMEASURE']._serialized_start=235
  _globals['_TENSORWITHMEASURE']._serialized_end=348
  _globals['_EMPTY']._serialized_start=350
  _globals['_EMPTY']._serialized_end=357
  _globals['_DISTRIBUTEDCLIENT']._serialized_start=360
  _globals['_DISTRIBUTEDCLIENT']._serialized_end=961
# @@protoc_insertion_point(module_scope)
