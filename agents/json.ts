import type { z } from 'zod';

export function schemaInstruction(schemaName: string, shape: string): string {
  return [
    `Return only valid JSON for ${schemaName}.`,
    'Do not wrap JSON in markdown.',
    'Do not add commentary outside JSON.',
    `Schema shape: ${shape}`,
  ].join(' ');
}

export function parseStructuredOutput<T>(
  raw: unknown,
  schema: z.ZodType<T>,
  fallback: T
): T {
  const text = stringifyModelContent(raw).trim();

  try {
    return schema.parse(JSON.parse(extractJson(text)));
  } catch {
    return fallback;
  }
}

function stringifyModelContent(raw: unknown): string {
  if (typeof raw === 'string') return raw;
  if (Array.isArray(raw)) {
    return raw
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item && typeof item === 'object' && 'text' in item) return String(item.text);
        return JSON.stringify(item);
      })
      .join('\n');
  }

  return String(raw ?? '');
}

function extractJson(text: string): string {
  const firstObject = text.indexOf('{');
  const lastObject = text.lastIndexOf('}');

  if (firstObject >= 0 && lastObject > firstObject) {
    return text.slice(firstObject, lastObject + 1);
  }

  return text;
}
