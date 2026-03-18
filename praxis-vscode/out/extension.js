"use strict";
/**
 * Praxis VS Code Extension
 *
 * Provides:
 *  - Diagnostics (inline squiggles) by running `praxis validate` on save
 *  - "Praxis: Validate File" command
 *  - "Praxis: Run File" command (runs in terminal)
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = require("vscode");
const cp = require("child_process");
// ── Diagnostics collection ────────────────────────────────────────────────────
let diagnosticCollection;
function activate(context) {
    diagnosticCollection = vscode.languages.createDiagnosticCollection('praxis');
    context.subscriptions.push(diagnosticCollection);
    // Validate on save
    context.subscriptions.push(vscode.workspace.onDidSaveTextDocument((doc) => {
        if (doc.languageId === 'praxis' || doc.fileName.endsWith('.px')) {
            const cfg = getConfig();
            if (cfg.validateOnSave) {
                validateDocument(doc);
            }
        }
    }));
    // Validate on open
    context.subscriptions.push(vscode.workspace.onDidOpenTextDocument((doc) => {
        if (doc.languageId === 'praxis' || doc.fileName.endsWith('.px')) {
            validateDocument(doc);
        }
    }));
    // Clear diagnostics when document is closed
    context.subscriptions.push(vscode.workspace.onDidCloseTextDocument((doc) => {
        diagnosticCollection.delete(doc.uri);
    }));
    // Command: Validate
    context.subscriptions.push(vscode.commands.registerCommand('praxis.validate', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor.');
            return;
        }
        validateDocument(editor.document, true);
    }));
    // Command: Run
    context.subscriptions.push(vscode.commands.registerCommand('praxis.run', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showInformationMessage('No active editor.');
            return;
        }
        runDocument(editor.document);
    }));
    // Validate any already-open .px documents on startup
    vscode.workspace.textDocuments.forEach((doc) => {
        if (doc.languageId === 'praxis' || doc.fileName.endsWith('.px')) {
            validateDocument(doc);
        }
    });
}
function deactivate() {
    diagnosticCollection?.dispose();
}
function getConfig() {
    const cfg = vscode.workspace.getConfiguration('praxis');
    return {
        executablePath: cfg.get('executablePath', 'praxis'),
        validateOnSave: cfg.get('validateOnSave', true),
        mode: cfg.get('mode', 'dev'),
    };
}
// ── Validation ────────────────────────────────────────────────────────────────
function validateDocument(doc, showSuccess = false) {
    const cfg = getConfig();
    const filePath = doc.fileName;
    const args = ['validate', filePath, '--mode', cfg.mode];
    cp.execFile(cfg.executablePath, args, { timeout: 15000 }, (err, stdout, stderr) => {
        diagnosticCollection.delete(doc.uri);
        if (!err) {
            if (showSuccess) {
                vscode.window.showInformationMessage('Praxis: ✓ Valid');
            }
            return;
        }
        const output = (stderr || stdout || '').trim();
        const diagnostics = parseDiagnostics(doc, output);
        diagnosticCollection.set(doc.uri, diagnostics);
        if (showSuccess || diagnostics.length > 0) {
            vscode.window.showWarningMessage(`Praxis: ${diagnostics.length} error(s). See Problems panel.`);
        }
    });
}
function runDocument(doc) {
    const cfg = getConfig();
    const filePath = doc.fileName;
    const terminal = vscode.window.createTerminal('Praxis Run');
    terminal.show();
    terminal.sendText(`${cfg.executablePath} run --file "${filePath}" --mode ${cfg.mode}`);
}
// ── Error parsing ─────────────────────────────────────────────────────────────
/**
 * Parse `praxis validate` stderr/stdout into VS Code Diagnostics.
 *
 * praxis validate outputs lines like:
 *   • UnknownVerb: BADVERB at line 3
 *   Parse error: Unexpected token at line 1 col 5
 *
 * We attempt to extract a line number; fall back to line 0.
 */
function parseDiagnostics(doc, output) {
    const diagnostics = [];
    // Try to find "line N" or "line N col M" in each error line
    const linePattern = /line\s+(\d+)(?:\s+col\s+(\d+))?/i;
    const bulletPattern = /^[•\-*]\s*/;
    const lines = output.split('\n').filter((l) => l.trim());
    for (const line of lines) {
        const cleaned = line.replace(bulletPattern, '').trim();
        if (!cleaned)
            continue;
        // Skip header lines like "Validation errors:" or "Validation failed:"
        if (/^(validation (errors|failed)|parse error:?\s*$)/i.test(cleaned))
            continue;
        const lineMatch = cleaned.match(linePattern);
        let lineNum = 0;
        let colNum = 0;
        if (lineMatch) {
            lineNum = Math.max(0, parseInt(lineMatch[1], 10) - 1);
            colNum = lineMatch[2] ? Math.max(0, parseInt(lineMatch[2], 10) - 1) : 0;
        }
        const range = new vscode.Range(new vscode.Position(lineNum, colNum), new vscode.Position(lineNum, doc.lineAt(Math.min(lineNum, doc.lineCount - 1)).text.length));
        const diag = new vscode.Diagnostic(range, cleaned, vscode.DiagnosticSeverity.Error);
        diag.source = 'praxis';
        diagnostics.push(diag);
    }
    // If we got output but parsed nothing, create a single generic error at line 0
    if (diagnostics.length === 0 && output.trim()) {
        const range = new vscode.Range(new vscode.Position(0, 0), new vscode.Position(0, doc.lineAt(0).text.length || 1));
        const diag = new vscode.Diagnostic(range, output.trim(), vscode.DiagnosticSeverity.Error);
        diag.source = 'praxis';
        diagnostics.push(diag);
    }
    return diagnostics;
}
//# sourceMappingURL=extension.js.map