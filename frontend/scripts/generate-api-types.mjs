import fs from "node:fs";
import path from "node:path";

const outputPath = path.resolve("./src/api/generated.ts");
const schemaPath = path.resolve("../docs/openapi.schema.json");

let schema = null;
if (fs.existsSync(schemaPath)) {
  schema = JSON.parse(fs.readFileSync(schemaPath, "utf-8"));
}

const paths = schema?.paths ?? {};
const routeInterfaces = Object.entries(paths)
  .map(([route, methods]) => {
    const safeRoute = JSON.stringify(route);
    const methodBlocks = Object.keys(methods)
      .map((method) => {
        return `    ${method}: { response: unknown };`;
      })
      .join("\n");

    return `  ${safeRoute}: {\n${methodBlocks}\n  };`;
  })
  .join("\n");

const generated = `/* eslint-disable */
// Auto-generated from OpenAPI schema by scripts/generate-api-types.mjs

import type { SessionCreateRequest, SessionListResponse, SessionRecord } from "../types/api";

export interface ApiPaths {
${routeInterfaces || "  '/api/sessions': {\n    get: { response: SessionListResponse };\n    post: { request: SessionCreateRequest; response: SessionRecord };\n  };"}
}

export type SessionListGet = SessionListResponse;
export type SessionCreatePost = SessionCreateRequest;
export type SessionCreateResponse = SessionRecord;
`;

fs.writeFileSync(outputPath, generated, "utf-8");
console.log(`Wrote ${outputPath}`);
