select distinct
    replaceRegexpAll(
        replaceRegexpAll(
            replaceRegexpAll(
                replaceRegexpAll(
                    replaceRegexpAll(
                        title,
                        '<[^>]+>', ''
                    ), '&[A-z]+dash;', '-'
                ), '&nbsp;', ' '
            ), '&[A-z]+quo;', '"'
        ), '&[A-z0-9]+;', ' '
    ) as txt
from raw.news
where text != '' and url not like '%/en/%'
order by published desc
limit 100000
