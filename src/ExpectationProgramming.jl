module ExpectationProgramming

# NOTE: Things don't work if we do "import Turing" because of some scoping 
# behaviour in the macros.
using Turing

import MacroTools

export @expectation,
    Expectation

struct Expectation
    gamma1_pos
    gamma1_neg
    gamma2
end

# TODO: Handle multiple return values, e.g. return a, b, c
macro expectation(expr)
    fn_dict = MacroTools.splitdef(expr)

    fn_name = fn_dict[:name]
    expct1_pos_name = gensym(fn_name)
    expct1_neg_name = gensym(fn_name)
    expct2_name = gensym(fn_name)

    fn_dict[:name] = expct2_name
    fn_expr_expct2 = MacroTools.combinedef(fn_dict)

    fn_body = fn_dict[:body]
    fn_dict[:body] = translate_return(fn_body, true)
    fn_dict[:name] = expct1_pos_name
    fn_expr_expct1_pos = MacroTools.combinedef(fn_dict)

    fn_dict[:body] = translate_return(fn_body, false)
    fn_dict[:name] = expct1_neg_name
    fn_expr_expct1_neg = MacroTools.combinedef(fn_dict)

    return quote
        esc(Turing.@model($fn_expr_expct1_pos))

        esc(Turing.@model($fn_expr_expct1_neg))

        esc(Turing.@model($fn_expr_expct2))

        $(esc(fn_name)) = Expectation(
            $(expct1_pos_name),
            $(expct1_neg_name),
            $(expct2_name)
        )
    end
end

function translate_return(expr, is_positive_expectation)
    MacroTools.postwalk(expr) do e
        if MacroTools.@capture(e, return r_)
            tmp_var = gensym()
            # NOTE: The compiler is smart enough to remove the if condition.
            # TODO: Possibly use the macro @addlogp!() to get rid of warning
            quote
                $(tmp_var) = $r
                if $is_positive_expectation
                    Turing.acclogp!(_varinfo, log(max($(tmp_var), 0)))
                else
                    Turing.acclogp!(_varinfo, log(-min($(tmp_var), 0)))
                end
                return $(tmp_var)
            end
        else
            e
        end
    end
end

end
