load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

package(default_visibility = ["//visibility:public"])

tf_cc_binary(
    name = "xla_compile",
    srcs = ["xla_compile.cc"],
    copts = [
        "-fexceptions",
    ],
    data = [
        "hlo_test.txt",
    ],
    deps = [
        ":hlo_graph",
        "//tensorflow/compiler/jit:xla_gpu_jit",
        "//tensorflow/compiler/xla:literal",
        "//tensorflow/compiler/xla:literal_util",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla:status",
        "//tensorflow/compiler/xla:statusor",
        "//tensorflow/compiler/xla/pjrt:cpu_device",
        "//tensorflow/compiler/xla/pjrt:gpu_device",
        "//tensorflow/compiler/xla/pjrt:pjrt_client",
        "//tensorflow/compiler/xla/service:hlo_pass_pipeline",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/tools:hlo_module_loader",
        "@com_github_gflags_gflags//:gflags",
    ],
)

tf_cc_binary(
    name = "graphviz",
    srcs = ["graphviz.cc"],
    copts = [
        "-fexceptions",
    ],
    deps = [
        "//tensorflow/compiler/xla/client:client_library",
        "//tensorflow/compiler/xla/client:local_client",
        "//tensorflow/compiler/xla/service:compiler",
        "//tensorflow/compiler/xla/service:cpu_plugin",
        "//tensorflow/compiler/xla/service:hlo_graph_dumper",
        "//tensorflow/compiler/xla/service:hlo_proto_cc",
        "//tensorflow/compiler/xla/service:hlo_runner",
        "//tensorflow/compiler/xla/service:local_service",
        "//tensorflow/compiler/xla/tools:hlo_extractor",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:protos_all_cc",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "hlo_graph",
    srcs = ["hlo_graph.cc"],
    hdrs = ["hlo_graph.h"],
    deps = [
        "//tensorflow/compiler/xla:types",
        "//tensorflow/compiler/xla:util",
        "//tensorflow/compiler/xla/service:hlo",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)
