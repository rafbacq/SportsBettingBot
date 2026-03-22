import React from 'react';

/**
 * Sport subcategory filter — shows only sport types (no category-level filter).
 * Dark themed with green accent chips.
 *
 * Props:
 *   activeSubcategory     — current subcategory (e.g., 'Basketball', null for all)
 *   onSubcategoryChange   — callback(subcategory)
 *   subcategoryCounts     — { Basketball: 80, Hockey: 20, ... }
 */
export default function FilterBar({
  activeSubcategory,
  onSubcategoryChange,
  subcategoryCounts = {},
}) {
  const subcategories = Object.keys(subcategoryCounts).sort((a, b) =>
    (subcategoryCounts[b] || 0) - (subcategoryCounts[a] || 0)
  );

  if (subcategories.length === 0) return null;

  return (
    <div className="filter-bar">
      <div className="filter-bar__row">
        <button
          className={`filter-chip ${!activeSubcategory ? 'filter-chip--active' : ''}`}
          onClick={() => onSubcategoryChange(null)}
        >
          All Sports
        </button>
        {subcategories.map((sub) => (
          <button
            key={sub}
            className={`filter-chip ${activeSubcategory === sub ? 'filter-chip--active' : ''}`}
            onClick={() => onSubcategoryChange(sub)}
          >
            {sub}
            <span className="filter-chip__count">{subcategoryCounts[sub]}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
