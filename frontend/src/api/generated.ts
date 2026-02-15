/* eslint-disable */
// Auto-generated from OpenAPI schema by scripts/generate-api-types.mjs

import type { SessionCreateRequest, SessionListResponse, SessionRecord } from "../types/api";

export interface ApiPaths {
  "/api/sessions": {
    get: {
      response: SessionListResponse;
    };
    post: {
      request: SessionCreateRequest;
      response: SessionRecord;
    };
  };
  "/api/sessions/{session_id}": {
    get: {
      response: SessionRecord;
    };
    delete: {
      response: { status: string };
    };
  };
  "/api/sessions/{session_id}/report": {
    get: {
      response: string;
    };
  };
  "/api/sessions/{session_id}/events": {
    get: {
      response: string;
    };
  };
}

export type SessionListGet = ApiPaths["/api/sessions"]["get"]["response"];
export type SessionCreatePost = ApiPaths["/api/sessions"]["post"]["request"];
export type SessionCreateResponse = ApiPaths["/api/sessions"]["post"]["response"];
