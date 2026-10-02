# API reference

The server speaks the Jev / SystemOne decision API. Clients written for Jev work unchanged.

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/systemone` | Answer typed questions about a state |
| `GET` | `/health` | `{"status": "ok", "model": ...}` once the model is loaded and warmed up |
| `GET` | `/v1/models` | The served model name |

If the server runs with `--api-key`, `POST /v1/systemone` requires `Authorization: Bearer <key>`.
`/health` stays open for load-balancer probes.

## Request

```json
{
  "model": "clef-flash",
  "state": "Our checkout started returning errors and orders are blocked.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle it?",
      "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"}
    },
    "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
    "outage": {"type": "noul", "instructions": "Is a service down?"}
  }
}
```

| Field | Required | Description |
|---|---|---|
| `model` | yes | Any string; echoed in the response |
| `state` | yes | Any JSON value: string, number, array, object or `null`. Truncated (not rejected) when the whole input exceeds the context window (16384 tokens for Clef-Flash) |
| `questions` | yes | Object of question ID to question, at least one |
| `images` | no | List of image references |
| `videos` | no | List of videos; each is a video reference or a list of image references (frames) |
| `media_kwargs` | no | Passed to the model's image/video processor, e.g. `{"images_kwargs": {"max_pixels": 200704}}` |

### Questions

| `type` | `criteria` | Limits | Answer |
|---|---|---|---|
| `choice` | Object of option ID to description (description may be `null`) | 1 to 255 options | One option |
| `score` | List of level descriptions, indexed from 0 | 2 to 10 levels | Expected level |
| `noul` | Optional `{"true": ..., "false": ...}` descriptions | Only keys `true` / `false` | Probability of true |

`instructions` is optional; the question ID is used when it is missing or empty.

### Images and videos

A media reference is any of:

- a data URI: `"data:image/png;base64,iVBORw0..."`
- raw base64: `"iVBORw0..."`
- an `http(s)://` URL, fetched by the server (disable with `--no-media-urls`)
- an object `{"url": "..."}`

A video is either one reference to a video file (decoded with PyAV, sampled uniformly to
`--video-frames` frames, 16 by default) or a list of image references taken as frames at
`--frame-list-fps` (2 by default). Every frame sent reaches the model. At most `--max-media-items`
(16) images and videos per request, each at most `--max-media-bytes` (50 MB).

## Response

```json
{
  "model": "clef-flash",
  "answers": {
    "department": {
      "type": "choice", "choice": "technical", "confidence": 0.9555,
      "probabilities": {"billing": 0.0445, "technical": 0.9555}
    },
    "urgency": {
      "type": "score", "score": 1.8209, "confidence": 0.8793,
      "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
      "probabilities": {"0": 0.0585, "1": 0.0622, "2": 0.8793}
    },
    "outage": {"type": "noul", "noul": 0.8105}
  },
  "usage": {"input_tokens": 299, "output_tokens": 0}
}
```

- `answers` keeps the order of `questions`; `probabilities` keeps the order of `criteria`.
- `choice`: `choice` is the most probable option and `confidence` its probability.
- `score`: `score` is the probability-weighted level, `confidence` the probability of the most likely level.
- `noul`: `noul` is the probability that the proposition is true.
- `usage.input_tokens` counts text and media tokens; `output_tokens` is always 0 (nothing is generated).

## Errors

Errors use one shape: `{"error": {"type": "...", "message": "..."}}`.

| Status | `type` | When |
|---|---|---|
| 400 | `invalid_request_error` | Malformed JSON, schema violation, undecodable media, schema longer than the context window |
| 401 | `authentication_error` | `--api-key` is set and the bearer token is missing or wrong |
| 503 | `overloaded_error` | GPU out of memory (transformers backend); retry with a smaller input |
| 500 | `api_error` | Anything else; logged with a traceback |
