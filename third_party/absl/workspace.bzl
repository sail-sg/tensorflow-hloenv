"""Provides the repository macro to import absl."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def repo():
    """Imports absl."""

    # Attention: tools parse and update these lines.
    # LINT.IfChange
    # ABSL_COMMIT = "215105818dfde3174fe799600bb0f3cae233d0bf"
    # ABSL_SHA256 = "237e2e6aec7571ae90d961d02de19f56861a7417acbbc15713b8926e39d461ed"
    ABSL_SHA256 = "520f61963f0807412d1d61f5f0dd706576b69413ee69959aa8f8715c49a78a00"
    ABSL_COMMIT = "731689ffc2ad7bb95cc86b5b6160dbe7858f27a0"
    # LINT.ThenChange(//tensorflow/lite/tools/cmake/modules/abseil-cpp.cmake)

    SYS_DIRS = [
        "algorithm",
        "base",
        "cleanup",
        "container",
        "debugging",
        "flags",
        "functional",
        "hash",
        "memory",
        "meta",
        "numeric",
        "random",
        "status",
        "strings",
        "synchronization",
        "time",
        "types",
        "utility",
    ]
    SYS_LINKS = {
        "//third_party/absl:system.absl.{name}.BUILD".format(name = n): "absl/{name}/BUILD.bazel".format(name = n)
        for n in SYS_DIRS
    }

    tf_http_archive(
        name = "com_google_absl",
        sha256 = ABSL_SHA256,
        build_file = "//third_party/absl:com_google_absl.BUILD",
        system_build_file = "//third_party/absl:system.BUILD",
        system_link_files = SYS_LINKS,
        patch_file = ["//third_party/absl:hloenv_make_hash_deterministic.patch"],
        strip_prefix = "abseil-cpp-{commit}".format(commit = ABSL_COMMIT),
        urls = tf_mirror_urls("https://github.com/abseil/abseil-cpp/archive/{commit}.tar.gz".format(commit = ABSL_COMMIT)),
    )
