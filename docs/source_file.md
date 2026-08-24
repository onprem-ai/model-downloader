# `.source.json` provenance format

## Purpose

Every published model directory contains a `.source.json` file describing where
its payload originated. Integrity and transfer state intentionally live
elsewhere:

```text
model-directory/
├── original model payload...
├── .source.json                  # durable source provenance
├── SHA256SUMS                    # inventory and SHA-256 integrity
└── SHA256SUMS.sigstore.json      # signature over SHA256SUMS
```

`.source.json` must never contain download URLs, credentials, authorization
headers, progress, ranges, retries, errors, local paths, or S3 storage identity.
Mutable transfer state belongs only in SQLite.

The normative machine-readable definition is
[`source-v1.schema.json`](../src/opai_models/schemas/source-v1.schema.json), using
JSON Schema Draft 2020-12. The schema is bundled in the installed Python package.

## Version 1 example

```json
{
  "schema_version": 1,
  "source": {
    "provider": "huggingface",
    "repository": "Sehyo/Qwen3.5-122B-A10B-NVFP4",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "subdirectory": null
  },
  "acquisition": {
    "acquired_at": "2026-08-24T09:00:00Z",
    "tool": {
      "name": "opai-model-publisher",
      "version": "1.0.0"
    }
  },
  "upstream_metadata": {
    "licenses": ["apache-2.0"],
    "library_name": "transformers",
    "pipeline_tag": "text-generation",
    "languages": ["en"],
    "tags": ["safetensors"]
  }
}
```

## Field semantics

- `schema_version` is required and is exactly `1`.
- `source.provider` is required and is exactly `huggingface` in version 1.
  Supporting another provider requires a new schema version or an explicitly
  defined additional source variant.
- `source.repository` is the Hugging Face `namespace/name` repository ID.
- `source.revision` is the immutable, lowercase, 40-character Git commit SHA
  resolved when the model was acquired.
- `source.subdirectory` is optional and identifies an upstream subdirectory when
  only that subtree was imported.
- `acquisition.acquired_at` records when this snapshot was acquired. It does not
  affect content identity and does not attest that the files remain unchanged.
- `acquisition.tool` optionally identifies the importing tool and version.
- `upstream_metadata` is an optional normalized snapshot of selected model-card
  fields. It is informational; consumers must not infer authorization,
  compatibility, or present license status from it alone.

All objects are closed: unknown properties fail validation. New meanings must be
introduced deliberately through a schema revision rather than arbitrary fields.

## Revision policy

New model publications must resolve tags and branches to an immutable Hugging
Face commit and write that commit to `source.revision`. Values such as `main`, a
tag, or a pull-request ref are not valid revisions.

The schema permits `null` only when the exact historical commit cannot be
established. Producers must not invent a revision. Application policy must
reject `null` for new publications.

## Serialization

Writers must emit UTF-8 JSON with:

- two-space indentation;
- LF line endings;
- one final newline;
- keys in the documented order where practical;
- no byte-order mark;
- no duplicate object keys;
- no non-finite numbers.

Consumers must rely on JSON field names rather than key order. `.source.json` is
provenance metadata and is not itself the model content identity.

## Relationship to `SHA256SUMS`

`SHA256SUMS` lists `.source.json` and every payload file. It excludes itself and
`SHA256SUMS.sigstore.json` to avoid recursion. The exact canonical bytes of
`SHA256SUMS` identify the complete model snapshot and are the bytes authenticated
by the Sigstore bundle. `.source.json` does not duplicate the file inventory or
checksums.

## Relationship to BagIt

The directory is not a BagIt bag and does not claim RFC 8493 conformance. An
exporter can translate `SHA256SUMS` into `manifest-sha256.txt`, copy payload files
under `data/`, and map selected `.source.json` fields to documented BagIt tags.
Importing a bag performs the reverse transformation. No `.bagit.json` file is
needed or defined.
