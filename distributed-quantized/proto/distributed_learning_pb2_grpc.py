# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from proto import distributed_learning_pb2 as proto_dot_distributed__learning__pb2

GRPC_GENERATED_VERSION = '1.65.5'
GRPC_VERSION = grpc.__version__
EXPECTED_ERROR_RELEASE = '1.66.0'
SCHEDULED_RELEASE_DATE = 'August 6, 2024'
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    warnings.warn(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in proto/distributed_learning_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
        + f' This warning will become an error in {EXPECTED_ERROR_RELEASE},'
        + f' scheduled for release on {SCHEDULED_RELEASE_DATE}.',
        RuntimeWarning
    )


class DistributedClientStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Initialize = channel.unary_unary(
                '/distributed_learning.DistributedClient/Initialize',
                request_serializer=proto_dot_distributed__learning__pb2.Dictionary.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.Empty.FromString,
                _registered_method=True)
        self.Forward = channel.unary_unary(
                '/distributed_learning.DistributedClient/Forward',
                request_serializer=proto_dot_distributed__learning__pb2.Query.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.Tensor.FromString,
                _registered_method=True)
        self.Backward = channel.unary_unary(
                '/distributed_learning.DistributedClient/Backward',
                request_serializer=proto_dot_distributed__learning__pb2.TensorWithLR.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.Query.FromString,
                _registered_method=True)
        self.GetModelState = channel.unary_unary(
                '/distributed_learning.DistributedClient/GetModelState',
                request_serializer=proto_dot_distributed__learning__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.ModelState.FromString,
                _registered_method=True)
        self.SetModelState = channel.unary_unary(
                '/distributed_learning.DistributedClient/SetModelState',
                request_serializer=proto_dot_distributed__learning__pb2.ModelState.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.Empty.FromString,
                _registered_method=True)
        self.GenerateQuantizedModel = channel.unary_unary(
                '/distributed_learning.DistributedClient/GenerateQuantizedModel',
                request_serializer=proto_dot_distributed__learning__pb2.Empty.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.Empty.FromString,
                _registered_method=True)
        self.TestInference = channel.unary_unary(
                '/distributed_learning.DistributedClient/TestInference',
                request_serializer=proto_dot_distributed__learning__pb2.Query.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.TensorWithMeasure.FromString,
                _registered_method=True)
        self.TestQuantizedInference = channel.unary_unary(
                '/distributed_learning.DistributedClient/TestQuantizedInference',
                request_serializer=proto_dot_distributed__learning__pb2.Query.SerializeToString,
                response_deserializer=proto_dot_distributed__learning__pb2.TensorWithMeasure.FromString,
                _registered_method=True)


class DistributedClientServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Initialize(self, request, context):
        """Initialize client
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Forward(self, request, context):
        """Training model
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Backward(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelState(self, request, context):
        """Aggregate model
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetModelState(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GenerateQuantizedModel(self, request, context):
        """Quantize model
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestInference(self, request, context):
        """Test model precision
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TestQuantizedInference(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_DistributedClientServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Initialize': grpc.unary_unary_rpc_method_handler(
                    servicer.Initialize,
                    request_deserializer=proto_dot_distributed__learning__pb2.Dictionary.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.Empty.SerializeToString,
            ),
            'Forward': grpc.unary_unary_rpc_method_handler(
                    servicer.Forward,
                    request_deserializer=proto_dot_distributed__learning__pb2.Query.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.Tensor.SerializeToString,
            ),
            'Backward': grpc.unary_unary_rpc_method_handler(
                    servicer.Backward,
                    request_deserializer=proto_dot_distributed__learning__pb2.TensorWithLR.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.Query.SerializeToString,
            ),
            'GetModelState': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelState,
                    request_deserializer=proto_dot_distributed__learning__pb2.Empty.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.ModelState.SerializeToString,
            ),
            'SetModelState': grpc.unary_unary_rpc_method_handler(
                    servicer.SetModelState,
                    request_deserializer=proto_dot_distributed__learning__pb2.ModelState.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.Empty.SerializeToString,
            ),
            'GenerateQuantizedModel': grpc.unary_unary_rpc_method_handler(
                    servicer.GenerateQuantizedModel,
                    request_deserializer=proto_dot_distributed__learning__pb2.Empty.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.Empty.SerializeToString,
            ),
            'TestInference': grpc.unary_unary_rpc_method_handler(
                    servicer.TestInference,
                    request_deserializer=proto_dot_distributed__learning__pb2.Query.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.TensorWithMeasure.SerializeToString,
            ),
            'TestQuantizedInference': grpc.unary_unary_rpc_method_handler(
                    servicer.TestQuantizedInference,
                    request_deserializer=proto_dot_distributed__learning__pb2.Query.FromString,
                    response_serializer=proto_dot_distributed__learning__pb2.TensorWithMeasure.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'distributed_learning.DistributedClient', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('distributed_learning.DistributedClient', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class DistributedClient(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Initialize(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/Initialize',
            proto_dot_distributed__learning__pb2.Dictionary.SerializeToString,
            proto_dot_distributed__learning__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Forward(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/Forward',
            proto_dot_distributed__learning__pb2.Query.SerializeToString,
            proto_dot_distributed__learning__pb2.Tensor.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Backward(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/Backward',
            proto_dot_distributed__learning__pb2.TensorWithLR.SerializeToString,
            proto_dot_distributed__learning__pb2.Query.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetModelState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/GetModelState',
            proto_dot_distributed__learning__pb2.Empty.SerializeToString,
            proto_dot_distributed__learning__pb2.ModelState.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def SetModelState(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/SetModelState',
            proto_dot_distributed__learning__pb2.ModelState.SerializeToString,
            proto_dot_distributed__learning__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GenerateQuantizedModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/GenerateQuantizedModel',
            proto_dot_distributed__learning__pb2.Empty.SerializeToString,
            proto_dot_distributed__learning__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def TestInference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/TestInference',
            proto_dot_distributed__learning__pb2.Query.SerializeToString,
            proto_dot_distributed__learning__pb2.TensorWithMeasure.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def TestQuantizedInference(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/distributed_learning.DistributedClient/TestQuantizedInference',
            proto_dot_distributed__learning__pb2.Query.SerializeToString,
            proto_dot_distributed__learning__pb2.TensorWithMeasure.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
