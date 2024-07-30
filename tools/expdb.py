import json
import os
import re

import click
import pandas as pd

"""
expdb.py: A CLI Tool for Managing and Viewing DataComp Language Models Project Data

This script, part of the DataComp Language Models project, serves as a command-line interface (CLI) tool for
managing and viewing experimental data related to datasets, model training, and evaluations. The project focuses
on creating datasets (stored on S3), training models, and evaluating them. The 'expdb' tool helps manage this data,
which is tracked using Git for metadata while pointing to S3 objects for actual data storage.

Directory Structure:
exp_data/
    datasets/
    models/
    evals/

Primary Objectives:
1. Maintain a clean and persistent specification for training/test set configurations from baselines to evals.
2. Ensure easy reproducibility of experiments across dataset creation, training, and evaluation phases.
3. Utilize JSON files for low-overhead interoperability and readability across different infrastructures.

The script provides functionality for:
- Reading JSON files representing datasets, models, and evaluation results.
- Merging data based on UUID references.
- Displaying the data in a tabular format with options for filtering, column selection, and output formatting.

Features:
- Filter data based on column conditions, including '=', '>', '>=', '<', '<=', and regex patterns.
- Output results to a searchable HTML file using DataTables for interactive capabilities.

Usage:
python tools/expdb.py --table models --output_html models.html

This example loads data from the 'models' table and outputs it to a searchable HTML file named 'models.html'.
This allows for easy viewing and sorting of model data, aiding in the analysis and reproducibility of experiments.

Reproducibility Contract:
- From a JSON file for a model, one can quickly reproduce the model using the 'open_lm_training_command'.
- Dataset JSON files enable tracing and reproducing the dataset creation process.
- Evaluation JSON files allow for exact re-running of evaluations to verify and replicate results.

Note:
- Ensure the provided path and table name are correct and exist in your file system.
- Internet connection is required for the DataTables functionality in the HTML output.
"""

# HTML template with DataTables for interactive tables
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data Table</title>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.css">
<script type="text/javascript" charset="utf8" src="https://code.jquery.com/jquery-3.3.1.js"></script>
<script type="text/javascript" charset="utf8" src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.js"></script>
</head>
<body>
{table}
<script>
$(document).ready(function() {{
    $('table').DataTable();
}});
</script>
</body>
</html>
"""


def load_smart_html(df):
    # Specify fixed columns
    fixed_columns = ["_source_json", "uuid", "dataset_name", "model", "params", "tokens", "low_variance_datasets"]

    df.rename(columns={"hyperparameters.params": "params", "hyperparameters.tokens": "tokens"}, inplace=True)
    # Generate column definitions for ag-Grid
    column_defs = [
        {
            "headerName": col.replace("eval_metrics.icl", "icl").replace("eval_metrics.downstream_perpexity", "ppl"),
            "field": col,
        }
        for col in df.columns
    ]
    col_widths = {"params": 100, "tokens": 100, "_source_json": 400}
    for column in column_defs:
        if column["field"] == "low_variance_datasets":
            column["sort"] = "desc"
        if column["field"] == fixed_columns[0]:
            column["pinned"] = "left"
        if column["field"] in col_widths:
            column["width"] = col_widths[column["field"]]
            column["initialWidth"] = col_widths[column["field"]]

    # Convert DataFrame rows to dictionaries for ag-Grid
    df = df.fillna("")
    df["_source_json"] = df["_source_json"].str.replace("exp_data/evals/", "")
    rows = df.to_dict(orient="records")

    # HTML template with ag-Grid setup
    html_template = f"""<!DOCTYPE html>
    <html lang="en" style="height: 100%; margin: 0">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DCNLP Results Viewer</title>
        <script src="https://cdn.jsdelivr.net/npm/ag-grid-community@31.3.2/dist/ag-grid-community.min.js"></script>
        <link href=" https://cdn.jsdelivr.net/npm/ag-grid-community@31.3.2/styles/ag-grid.min.css " rel="stylesheet">
        <style type='text/css'>
        .ag-theme-quartz {{
            --ag-grid-size: 5px;
            --ag-list-item-height: 20px;
        }}
        #code {{
            display: none;
            margin-left: 20px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }}
        .right-align {{
            margin-left: auto;
        }}
        </style>
    </head>
    <body style="height: 100%; margin: 0">
        <div style="display: flex; flex-direction: column; height: 100%">
        <div class='header'>
        <span>
        Column Search: <input type="text" id="colSearch" placeholder="Type to filter columns..." oninput="onColSearchInput()">
        </span>
        <span class='right-align'>
        <button onClick="toggleLoaderCode()">Show dataframe code</button>
        </span>
        </div>
        <div id="code"></div>
        <div id="myGrid" class="ag-theme-quartz" style="width:100%; flex-grow: 1"></div>
        </div>
        <script>
            function toggleLoaderCode() {{
                var codeDiv = document.getElementById('code');
                if (codeDiv.style.display === 'none' || codeDiv.style.display === '') {{
                    codeDiv.style.display = 'block'; // Show the div
                    // Assuming your ag-Grid instance is initialized and available as 'gridOptions'
                    var rows = [];

                    // Access all displayed (visible) rows in the grid
                    gridOptions.api.forEachNodeAfterFilterAndSort(function(node) {{
                        // Check if the row node is visible
                        if (!node.group && node.data) {{ // Ensure it's not a group node and data exists
                            rows.push(`'${{node.data.uuid}}'`);
                        }}
                    }});
                    codeDiv.innerHTML = (
                        `<pre><code>`
                        + `uuids = [${{rows.join(', ')}}]\n`
                        + `from tools.expdb import build_table_dfs, enrich_evals\n`
                        + `full_df = enrich_evals(build_table_dfs("exp_data"));\n`
                        + `df = full_df.set_index("uuid").loc[uuids].reset_index()`
                        + `</code></pre>`
                    );
                }} else {{
                    codeDiv.style.display = 'none'; // Hide the div
                }}
            }}

            function humanReadable(num) {{
                if (num === undefined || num === null) {{
                    return null;
                }} else if (num >= 1_000_000_000_000) {{
                    return (num / 1_000_000_000_000).toFixed(1) + 'T';
                }} else if (num >= 1_000_000_000) {{
                    return (num / 1_000_000_000).toFixed(1) + 'B';
                }} else if (num >= 1_000_000) {{
                    return (num / 1_000_000).toFixed(1) + 'M';
                }} else if (num >= 1_000) {{
                    return (num / 1_000).toFixed(1) + 'K';
                }} else {{
                    return num.toString();
                }}
            }}
            var columnDefs = {json.dumps(column_defs)};
            let regexColumns = ["_source_json", "uuid", "name", "dataset_name", "model_name"];
            let numberColumns = ["params", "tokens"];
            columnDefs.forEach(column => {{
                if (numberColumns.includes(column["field"])) {{
                    column["valueFormatter"] = params => humanReadable(params.value);
                    column["filter"] = "agNumberColumnFilter";
                    column["filterParams"] = {{
                        "maxNumConditions": 1
                    }}
                }}
                if (regexColumns.includes(column["field"])) {{
                    column["filterParams"] = {{
                        "filterOptions": [
                            {{
                            displayKey: 'regexp',
                            displayName: 'Regular Expression',
                            predicate: ([fv1], cellValue) => cellValue == null || new RegExp(fv1, 'gi').test(cellValue),
                            numberOfInputs: 1,
                            }}
                        ]
                    }}
                }}
            }});

            var rowData = {json.dumps(rows)};

            function onColSearchInput() {{
                updateUrlWithCurrentFilterState();
                filterColumns();
            }}

            function filterColumns() {{
                var searchValue = document.getElementById('colSearch').value.toLowerCase();
                try {{
                    var regex = new RegExp(searchValue, 'i');
                    var fixedColumns = {fixed_columns};
                    var filteredColumns = columnDefs.filter(function(column) {{
                        return regex.test(column.headerName) || fixedColumns.includes(column.field);
                    }});
                    // Ensure fixed columns are always at the beginning
                    var orderedFilteredColumns = fixedColumns.map(fc => filteredColumns.find(c => c.headerName === fc)).concat(
                        filteredColumns.filter(c => !fixedColumns.includes(c.headerName))
                    );
                    gridOptions.api.setColumnDefs(orderedFilteredColumns);
                }} catch(e) {{
                    console.error('Invalid regex', e);
                }}
            }}

            var gridOptions = {{
                columnDefs: columnDefs,
                rowData: rowData,
                defaultColDef: {{
                    resizable: true,
                    filter: true,
                    sortable: true,
                    floatingFilter: true,
                }},
                enableCellTextSelection: true,
                ensureDomOrder: true,
                suppressFieldDotNotation: true,
            }};

            function updateUrlWithCurrentFilterState() {{
                const allFilterModel = gridOptions.api.getFilterModel();
                const columnSearchValue = document.getElementById('colSearch').value;
                const stateObject = {{
                    filterModel: allFilterModel,
                    columnSearch: columnSearchValue
                }};

                const stateAsString = JSON.stringify(stateObject);
                const encodedFilterState = encodeURIComponent(stateAsString);
                // Assuming you're using the hash fragment to store the state
                window.location.hash = 'state=' + encodedFilterState;
            }}

            function applyFiltersFromUrl() {{
                const hash = window.location.hash.substring(1); // Remove '#'
                const params = new URLSearchParams(hash);
                let state = null;
                if (params.has('filters')) {{ // Old version of params
                    state = {{'filterModel': JSON.parse(decodeURIComponent(params.get('filters')))}}
                }} else {{
                    const stateAsString = params.get('state');
                    if (stateAsString) {{
                        state = JSON.parse(decodeURIComponent(stateAsString));
                    }}
                }}
                if (state) {{
                    // Apply the filter model to ag-Grid
                    if (state.filterModel) {{
                        gridOptions.api.setFilterModel(state.filterModel);
                    }}

                    // Set the column search input's value and trigger the search
                    if (state.columnSearch !== undefined) {{
                        document.getElementById('colSearch').value = state.columnSearch;
                        filterColumns(); // Assuming filterColumns() applies the column search
                    }}
                }}
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                var gridDiv = document.getElementById('myGrid');
                new agGrid.Grid(gridDiv, gridOptions);
                filterColumns();
                // Apply filters from URL if present
                applyFiltersFromUrl();

                // Hook into ag-Grid's filter changed event
                gridOptions.api.addEventListener('filterChanged', updateUrlWithCurrentFilterState);
            }});
        </script>
    </body>
    </html>"""
    return html_template


def load_json_to_df(file_path, partition_keys):
    with open(file_path, "r") as file:
        df = pd.json_normalize(json.load(file))
        df["_source_json"] = file_path
        for idx, key in enumerate(partition_keys, start=1):
            df[f"partition_key_{idx}"] = key
        return df


def build_table_dfs(database_path, table=None):
    table_dfs = {}
    for table_name in os.listdir(database_path):
        if table is not None and table_name != table:
            continue  # save redundant loading time
        table_path = os.path.join(database_path, table_name)
        if os.path.isdir(table_path):
            dataframes = []
            for root, dirs, files in os.walk(table_path):
                partition_keys = os.path.relpath(root, table_path).split(os.sep)
                partition_keys = [key for key in partition_keys if key != "."]
                for file in files:
                    if not file.endswith("json"):
                        continue
                    file_path = os.path.join(root, file)
                    df = load_json_to_df(file_path, partition_keys)
                    dataframes.append(df)
            if len(dataframes) > 0:
                table_dfs[table_name] = pd.concat(dataframes, ignore_index=True)
    return table_dfs


def merge_uuid_references(table_dfs):
    for table_name, df in table_dfs.items():
        uuid_cols = [col for col in df.columns if col.endswith("_uuid")]
        for col in uuid_cols:
            # Adjust for singular to plural mapping
            ref_table_name = col.rsplit("_", 1)[0] + "s"
            if ref_table_name in table_dfs:
                ref_df = table_dfs[ref_table_name]
                if "name" in ref_df.columns:
                    # Merge and keep only the name from the reference table
                    merged_df = df.merge(ref_df[["uuid", "name"]], left_on=col, right_on="uuid", how="left")
                    df[f"{ref_table_name[:-1]}_name"] = merged_df["name_y"]
                    # Drop the original uuid column to avoid confusion
                    df.drop(columns=[col], inplace=True)
            table_dfs[table_name] = df
    return table_dfs


# Function to convert shorthand notations to float values
def convert_shorthand_to_number(s):
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    if s[-1] in multipliers:
        return float(s[:-1]) * multipliers[s[-1]]
    return float(s)


def filter_df(df, filter_str):
    for f in filter_str:
        if "=" in f and ">" not in f and "<" not in f:
            parts = f.split("=", 1)
            column_name, pattern = parts
            print(df)
            print(column_name, pattern)
            df = df[df[column_name].astype(str).str.contains(pattern, na=False)]
        else:
            parts = re.split("(>=|<=|>|<)", f, maxsplit=1)
            if len(parts) == 3:
                column_name, operator, threshold_str = parts
                threshold = convert_shorthand_to_number(threshold_str)

                if operator == ">":
                    df = df[df[column_name].astype(float) > threshold]
                elif operator == ">=":
                    df = df[df[column_name].astype(float) >= threshold]
                elif operator == "<":
                    df = df[df[column_name].astype(float) < threshold]
                elif operator == "<=":
                    df = df[df[column_name].astype(float) <= threshold]
            else:
                print(f"Invalid filter format: {f}")
    return df


def enrich_evals(table_dfs, additional_model_keys=None):
    # Enrich table with some extra information for evals.
    #   - # tokens, # params
    #   - chinchilla multiplier
    #   - datasets
    datasets_df = table_dfs["datasets"].set_index("uuid")
    models_df = table_dfs["models"].set_index("uuid")
    evals_df = table_dfs["evals"].set_index("uuid")
    model_keys = ["hyperparameters.tokens", "hyperparameters.params", "dataset_name", "dataset_uuid"]
    if additional_model_keys:
        model_keys += additional_model_keys
    evals_df = pd.merge(
        left=evals_df,
        right=models_df[model_keys],
        how="left",
        left_on="model_uuid",
        right_index=True,
    )
    evals_df = pd.merge(
        left=evals_df, right=datasets_df[["tokenizer"]], how="left", left_on="dataset_uuid", right_index=True
    )
    evals_df = evals_df[~evals_df.index.duplicated(keep="first")]
    return evals_df.reset_index()


@click.command()
@click.option("--database_path", default="exp_data")
@click.option("--table", default="models")
@click.option("--columns", "-c", multiple=True, help="Columns to display")
@click.option("--leading_columns", "-l", multiple=True, help="Columns to display")
@click.option("--max_rows", default=None, help="Max number of rows to display")
@click.option("--max_colwidth", default=50, help="Maximum width of each column")
@click.option("--filter", "-f", multiple=True, help="Filter rows based on column conditions")
@click.option("--output_html", default=None, help="Path to output HTML file")
@click.option("--output_csv", default=None, help="Path to output CSV file")
def main(
    database_path="exp_data",
    table="models",
    columns=None,
    leading_columns=None,
    max_rows=None,
    max_colwidth=50,
    filter=None,
    output_html=None,
    output_csv=None,
):
    table_dfs = build_table_dfs(database_path)

    if table == "evals":
        df = enrich_evals(table_dfs)
    else:
        merged_dfs = merge_uuid_references(table_dfs)
        df = merged_dfs[table]

    # Set the maximum column width
    pd.set_option("display.max_colwidth", max_colwidth)
    # Implementing enhanced filtering logic
    if filter:
        df = filter_df(df, filter)

    if columns:  # If specific columns are provided
        df = df[list(set(columns))]

    if leading_columns:
        all_cols = df.columns
        last_cols = []
        lead_cols = []
        for c in leading_columns:
            for c2 in all_cols:
                c2_ser = pd.Series(c2)
                if c2_ser.str.contains(c).all() and c2 not in lead_cols:
                    lead_cols.append(c2)

            for c2 in all_cols:
                if c2 not in lead_cols:
                    last_cols.append(c2)

        df = df[lead_cols + last_cols]
    print(df)
    if output_html:
        html = load_smart_html(df)
        # Output to HTML file
        with open(output_html, "w") as f:
            f.write(html)
        print(f"DataFrame written to HTML file: {output_html}")

    if output_csv:
        df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
